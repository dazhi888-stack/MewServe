"""导入 FAQ 知识库到 Chroma（可独立运行，也可由服务启动时自动触发）。

用法：
    python scripts/ingest_faq.py            # 使用 .env 与默认路径
    python scripts/ingest_faq.py 路径/faq.md # 指定 FAQ 文件
"""

import sys
from pathlib import Path

# 让脚本可以从仓库根目录直接运行。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.rag import get_retriever  # noqa: E402


def main() -> None:
    settings = get_settings()
    faq_path = sys.argv[1] if len(sys.argv) > 1 else settings.faq_file
    retriever = get_retriever(settings)
    retriever.ensure_collection()
    count = retriever.ingest(faq_path)
    print(f"导入完成：{count} 条 FAQ 已写入 {settings.chroma_dir}")


if __name__ == "__main__":
    main()
