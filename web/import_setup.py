"""
导入助手 - 解决 Railway 部署时的 Python 路径问题
统一使用根目录的 app/ 包，消除 web/app/ 代码重复
"""
import sys
from pathlib import Path

def setup_paths():
    """确保所有必要的路径都在 sys.path 中"""
    # 指向项目根目录（web/ 的父目录），而非 web/ 自身
    project_root = Path(__file__).resolve().parent.parent

    # 添加项目根目录
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 检查并报告路径信息
    app_dir = project_root / "app"
    print(f"[IMPORT_SETUP] Project root: {project_root}", flush=True)
    print(f"[IMPORT_SETUP] App directory: {app_dir}", flush=True)
    print(f"[IMPORT_SETUP] App directory exists: {app_dir.exists()}", flush=True)

    if app_dir.exists():
        init_file = app_dir / "__init__.py"
        print(f"[IMPORT_SETUP] __init__.py exists: {init_file.exists()}", flush=True)

    return project_root

if __name__ == "__main__":
    setup_paths()
