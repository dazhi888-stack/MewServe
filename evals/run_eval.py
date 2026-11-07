"""离线评估管线：意图分类准确率/混淆矩阵 + RAG 检索 recall@k/MRR。

用法（在项目根目录，需已配置 .env）：
    uv run python evals/run_eval.py                 # 全量评估
    uv run python evals/run_eval.py --part intent   # 只评估意图
    uv run python evals/run_eval.py --part retrieval
    uv run python evals/run_eval.py --limit 10      # 快速抽样

产出：evals/report.md（含时间戳、配置快照与各项指标）。
"""

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.model import create_model  # noqa: E402
from app.rag import get_retriever  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_qa.jsonl"
REPORT_PATH = Path(__file__).resolve().parent / "report.md"


def load_golden(limit: int | None) -> list[dict]:
    entries = []
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if limit:
        entries = entries[:limit]
    return entries


def _title(document: str) -> str:
    return document.split("\n", 1)[0]


# ---------- 意图评估 ----------

def eval_intent(entries: list[dict], settings, model) -> dict:
    from app.intent import classify_intent

    async def run():
        results = []
        for entry in entries:
            try:
                intent = await classify_intent(entry["question"], settings, model)
                results.append((entry, intent.category))
            except Exception as exc:  # 单条失败不阻断评估
                results.append((entry, f"<error:{type(exc).__name__}>"))
        return results

    results = asyncio.run(run())
    correct = sum(1 for entry, pred in results if pred == entry["intent"])
    accuracy = correct / len(results) if results else 0.0

    # 混淆矩阵：仅统计真实标签与预测中出现过的类目。
    labels = sorted({entry["intent"] for entry, _ in results} |
                    {pred for _, pred in results if not pred.startswith("<error")})
    matrix: dict[str, Counter] = {label: Counter() for label in labels}
    for entry, pred in results:
        matrix[entry["intent"]][pred] += 1

    # 每类 precision/recall/f1。
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        pred_total = sum(matrix[other][label] for other in labels)
        true_total = sum(matrix[label].values())
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / true_total if true_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = (precision, recall, f1, true_total)

    errors = [(entry["id"], entry["intent"], pred)
              for entry, pred in results if pred != entry["intent"]]
    return {"accuracy": accuracy, "labels": labels, "matrix": matrix,
            "per_class": per_class, "errors": errors, "total": len(results)}


# ---------- 检索评估 ----------

def eval_retrieval(entries: list[dict], settings, retriever) -> dict:
    retriever.ensure_collection()
    if not retriever.available:
        raise SystemExit("Chroma 知识库不可用，无法评估检索。")

    ks = [1, 3, 5]
    hits = {k: 0 for k in ks}
    reciprocal_ranks = []
    evaluated = 0
    misses = []
    for entry in entries:
        expected = entry.get("expect_faq") or []
        if not expected:
            continue
        evaluated += 1
        docs = retriever.retrieve(entry["question"])
        titles = [_title(doc) for doc in docs]
        rank = None
        for position, title in enumerate(titles, start=1):
            if any(keyword in title for keyword in expected):
                rank = position
                break
        if rank is None:
            misses.append((entry["id"], entry["question"], [t[:30] for t in titles[:3]]))
        else:
            reciprocal_ranks.append(1.0 / rank)
            for k in ks:
                if rank <= k:
                    hits[k] += 1

    recall = {k: hits[k] / evaluated if evaluated else 0.0 for k in ks}
    mrr = sum(reciprocal_ranks) / evaluated if evaluated else 0.0
    return {"evaluated": evaluated, "recall": recall, "mrr": mrr,
            "misses": misses, "top_k": settings.rag_top_k}


# ---------- 报告 ----------

def write_report(settings, intent_result, retrieval_result) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# MewHelp 离线评估报告（{now}）",
        "",
        f"- 模型：`{settings.deepseek_model}` | 检索：向量+BM25 混合，"
        f"rerank={'on' if settings.rerank_enabled else 'off'}，top_k={settings.rag_top_k}",
        "",
    ]
    if intent_result is not None:
        lines += [
            "## 意图分类",
            "",
            f"- 样本数：{intent_result['total']}，准确率：**{intent_result['accuracy']:.1%}**",
            "",
            "| 类目 | Precision | Recall | F1 | 支持数 |",
            "|---|---|---|---|---|",
        ]
        for label, (p, r, f1, support) in intent_result["per_class"].items():
            lines.append(f"| {label} | {p:.2f} | {r:.2f} | {f1:.2f} | {support} |")
        if intent_result["errors"]:
            lines += ["", "### 误分类明细", ""]
            lines += [f"- {eid}: {true} → {pred}"
                      for eid, true, pred in intent_result["errors"]]
        lines.append("")
    if retrieval_result is not None:
        lines += [
            "## FAQ 检索",
            "",
            f"- 样本数：{retrieval_result['evaluated']}（最终 top_k={retrieval_result['top_k']}）",
            f"- Recall@1：**{retrieval_result['recall'][1]:.1%}**",
            f"- Recall@3：**{retrieval_result['recall'][3]:.1%}**",
            f"- Recall@5：**{retrieval_result['recall'][5]:.1%}**",
            f"- MRR：**{retrieval_result['mrr']:.3f}**",
        ]
        if retrieval_result["misses"]:
            lines += ["", "### 未命中明细", ""]
            lines += [f"- {eid}「{q}」→ {titles}"
                      for eid, q, titles in retrieval_result["misses"]]
        lines.append("")
    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="MewHelp 离线评估")
    parser.add_argument("--part", choices=["all", "intent", "retrieval"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 条标注")
    args = parser.parse_args()

    settings = get_settings()
    entries = load_golden(args.limit)
    print(f"加载 {len(entries)} 条标注样本（part={args.part}）")

    intent_result = None
    retrieval_result = None
    model = None
    if args.part in ("all", "intent"):
        if not settings.model_configured:
            raise SystemExit("DEEPSEEK_* 未配置，无法评估意图。")
        model = create_model(settings)
        print("评估意图分类中……（每条一次模型调用，请耐心等待）")
        intent_result = eval_intent(entries, settings, model)
        print(f"意图准确率：{intent_result['accuracy']:.1%} "
              f"({intent_result['total']} 条)")
    if args.part in ("all", "retrieval"):
        retriever = get_retriever(settings)
        print("评估 FAQ 检索中……")
        retrieval_result = eval_retrieval(entries, settings, retriever)
        print(f"Recall@1/3/5：{retrieval_result['recall'][1]:.1%} / "
              f"{retrieval_result['recall'][3]:.1%} / {retrieval_result['recall'][5]:.1%}"
              f" | MRR：{retrieval_result['mrr']:.3f}")

    report = write_report(settings, intent_result, retrieval_result)
    print(f"\n报告已写入 {REPORT_PATH}\n")
    print(report)


if __name__ == "__main__":
    main()
