"""支持 python3 -m compute_network_scheduler 调用命令行。"""

from .interface.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
