"""Cho phép chạy dự án bằng ``python -m src``."""

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
