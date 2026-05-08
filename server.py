"""
暖暖记忆助手 - 统一服务端入口
启动 Flask 后端服务，桌面端和网页端共用
"""
import argparse
import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 设置默认数据目录（根目录下的 data/）
os.environ.setdefault("DATA_DIR", str(project_root / "data"))
os.environ.setdefault("VAULT_DIR", str(project_root / "data" / "memory_vault"))

# 导入 Flask 应用（web/main.py 中的 app）
from web.main import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="暖暖记忆助手 - 服务端")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="监听端口 (默认 5000)")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    data_dir = os.environ.get("DATA_DIR", str(project_root / "data"))
    print(f"暖暖 - 记忆助手服务端", file=sys.stderr, flush=True)
    print(f"  地址: {args.host}:{args.port}", file=sys.stderr, flush=True)
    print(f"  数据: {data_dir}", file=sys.stderr, flush=True)
    print(f"  API:  http://{args.host}:{args.port}/api/health", file=sys.stderr, flush=True)

    app.run(host=args.host, port=args.port, debug=args.debug)
