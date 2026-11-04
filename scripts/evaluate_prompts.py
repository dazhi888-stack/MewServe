"""Bounded synthetic prompt evaluation; never stores credentials or raw errors."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
from time import perf_counter

# `python scripts/evaluate_prompts.py` puts scripts/, not the project root,
# on sys.path. Resolve independently of cwd before importing the application.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.context import build_chat_messages
from app.errors import AppError
from app.model import create_model
from app.schemas import ChatRequest
from app.service import extract_after_sales, stream_chat


async def evaluate(settings, report):
    for kind in ("extraction", "customer_service"):
        for item in report[kind]:
            start = perf_counter()
            model = create_model(settings)
            try:
                if kind == "extraction":
                    actual = (await extract_after_sales(item["description"], settings, model)).model_dump()
                    item["actual"] = actual
                    item["field_scores"] = {key: int(actual[key] == value) for key, value in item["expected"].items()}
                    item["score_method"] = "exact_match; expected_solution paraphrases require manual semantic review"
                else:
                    prepared = build_chat_messages(ChatRequest(message=item["message"], history=item["history"]), settings)
                    answer = []
                    async for event, data in stream_chat(prepared, model):
                        if event == "token":
                            answer.append(data["text"])
                        elif event == "error":
                            raise AppError(data["code"], data["message"], 502)
                    item["actual"] = "".join(answer)
                    item["rubric_assessment"] = "pending_manual_review"
                item["status"] = "executed"
            except AppError as exc:
                item["status"] = "failed"
                item["error_code"] = exc.code
                report["status"] = "failed"
                return 1
            finally:
                item["elapsed_seconds"] = round(perf_counter() - start, 4)
                await model.root_async_client.close()
    report["status"] = "executed_pending_semantic_review"
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "not_executed", "synthetic_inputs": True,
        "extraction": json.loads((ROOT / "evals/aftersales.json").read_text(encoding="utf-8")),
        "customer_service": json.loads((ROOT / "evals/customer_service.json").read_text(encoding="utf-8")),
        "limitations": "Exact field matching is diagnostic; chat rubrics and paraphrased solutions need human review. This is not chapter acceptance.",
    }
    for kind in ("extraction", "customer_service"):
        for item in report[kind]:
            item["status"] = "not_executed"
    try:
        settings = Settings()
        report["model"] = settings.openai_model or None
        report["structured_output_method"] = settings.structured_output_method
        for kind in ("extraction", "customer_service"):
            for item in report[kind]:
                item["model"] = report["model"]
        if not settings.model_configured:
            report["reason"] = "model_not_configured"
            result = 2
        else:
            result = asyncio.run(evaluate(settings, report))
    except Exception:
        # Do not print configuration validation values, SDK payloads or errors.
        report["status"] = "failed"
        report["reason"] = "evaluation_failed"
        result = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"evaluation: {report['status']}; output: {args.output}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
