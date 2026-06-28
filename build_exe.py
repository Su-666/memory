"""
暖暖记忆助手 - 一键打包脚本
将 Flask + pywebview 打包为 Windows 安装程序。

流程：PyInstaller 打包 → Inno Setup 编译安装包 → 清理中间产物

使用方式：
    pip install pyinstaller pywebview pythonnet pywin32
    python build_exe.py

最终输出：
    installer_output/暖暖记忆助手-Setup.exe
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 强制 Windows 控制台使用 UTF-8 编码，避免中文路径报 UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

APP_NAME = "暖暖记忆助手"
ISS_FILE = "安装包.iss"
VERSION_FILE = "VERSION"


def read_version(root: Path) -> str:
    """读取 VERSION 文件中的版本号"""
    p = root / VERSION_FILE
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return "6.0.0"


def bump_version(root: Path) -> str:
    """版本号 patch 位自动 +1（如 6.0.0 → 6.0.1），返回新版本号"""
    current = read_version(root)
    parts = current.split(".")
    # 补齐到三段：major.minor.patch
    while len(parts) < 3:
        parts.append("0")
    try:
        parts[2] = str(int(parts[2]) + 1)
    except ValueError:
        parts[2] = "1"
    new_version = ".".join(parts[:3])
    (root / VERSION_FILE).write_text(new_version + "\n", encoding="utf-8")
    return new_version


def check_dependency(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def find_iscc() -> str | None:
    """查找 Inno Setup 编译器 ISCC.exe 路径"""
    # 1. PATH 中查找（含 winget 安装的用户目录）
    iscc = shutil.which("ISCC.exe") or shutil.which("ISCC")
    if iscc:
        return iscc
    # 2. 常见安装路径（winget / 手动安装）
    candidates = []
    localappdata = os.environ.get("LOCALAPPDATA", "")
    if localappdata:
        candidates.append(os.path.join(localappdata, "Programs", "Inno Setup 6", "ISCC.exe"))
    candidates += [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    return None


def run_pyinstaller(root: Path) -> bool:
    """第一步：用 PyInstaller 打包到 dist/ 目录"""
    app_dir = root / "app"
    web_dir = root / "web"
    env_example = root / ".env.example"

    if not app_dir.exists() or not web_dir.exists():
        print("[错误] 找不到 app/ 或 web/ 目录，请在项目根目录运行此脚本")
        return False

    sep = ";" if sys.platform == "win32" else ":"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name", APP_NAME,
        "--windowed",            # 无控制台窗口
        "--onedir",              # 目录模式（启动快）
        "--icon", str(root / "appicon.ico"),   # 应用图标
        str(root / "launcher.py"),
        "--add-data", f"{app_dir}{sep}app",
        "--add-data", f"{web_dir}{sep}web",
    ]

    if env_example.exists():
        cmd += ["--add-data", f"{env_example}{sep}."]

    # VERSION 文件（程序运行时读取版本号）
    version_file = root / VERSION_FILE
    if version_file.exists():
        cmd += ["--add-data", f"{version_file}{sep}."]

    # 安全检查：确保不会把含真实密钥的 .env 打包进去
    real_env = root / ".env"
    if real_env.exists():
        print(f"[安全] 检测到本地 .env，已确保不打包到安装程序中")

    # 应用图标（pywebview 运行时使用）
    icon_file = root / "appicon.ico"
    if icon_file.exists():
        cmd += ["--add-data", f"{icon_file}{sep}."]

    # certifi 证书（HTTPS 请求需要）
    try:
        import certifi
        cacert = Path(certifi.where())
        if cacert.exists():
            cmd += ["--add-data", f"{cacert}{sep}certifi"]
    except ImportError:
        pass

    # hidden imports
    hidden = [
        "flask", "flask_cors", "flask_compress", "werkzeug", "jinja2",
        "itsdangerous", "click", "markupsafe",
        "app.db", "app.repo", "app.search", "app.answer",
        "app.intent_chat", "app.llm", "app.vision", "app.vault",
        "app.zhipu_client", "app.utils",
        "web.main", "web.import_setup",
        "main", "import_setup",
        "certifi", "chardet", "PIL", "PIL.Image", "aip",
        "webview", "webview.platforms.edgechromium", "webview.platforms.winforms",
        "clr_loader", "clr",
    ]
    for h in hidden:
        cmd += ["--hidden-import", h]

    cmd += ["--collect-submodules", "app"]
    cmd += ["--collect-submodules", "webview"]

    print("=" * 60)
    print("[1/3] PyInstaller 打包中...")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=str(root))
    return result.returncode == 0


def run_inno_setup(root: Path, version: str) -> bool:
    """第二步：用 Inno Setup 编译安装包"""
    iscc = find_iscc()
    if not iscc:
        print("[错误] 未找到 Inno Setup 编译器 ISCC.exe")
        print("请安装 Inno Setup 6: https://jrsoftware.org/isdl.php")
        return False

    iss_path = root / ISS_FILE
    if not iss_path.exists():
        print(f"[错误] 未找到安装脚本: {ISS_FILE}")
        return False

    print("\n" + "=" * 60)
    print("[2/3] Inno Setup 编译安装包中...")
    print(f"  ISCC: {iscc}")
    print(f"  版本: {version}")
    print("=" * 60)
    # 通过环境变量 APP_VERSION 把版本号传给 ISS 脚本
    env = os.environ.copy()
    env["APP_VERSION"] = version
    result = subprocess.run([iscc, str(iss_path)], cwd=str(root), env=env)
    return result.returncode == 0


def cleanup(root: Path) -> None:
    """第三步：清理中间产物"""
    print("\n" + "=" * 60)
    print("[3/3] 清理中间产物...")
    print("=" * 60)

    # 清理 dist/ 目录（PyInstaller 产物，已打包进安装包）
    dist_dir = root / "dist"
    if dist_dir.exists():
        shutil.rmtree(dist_dir, ignore_errors=True)
        print("  已删除 dist/")

    # 清理 build/ 目录（PyInstaller 缓存）
    build_dir = root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
        print("  已删除 build/")

    # 清理自动生成的 .spec 文件
    for spec in [root / f"{APP_NAME}.spec", root / "MemoryAssistant.spec"]:
        if spec.exists():
            spec.unlink()
            print(f"  已删除 {spec.name}")


def main() -> None:
    root = Path(__file__).resolve().parent

    # 依赖检查
    missing = []
    if not check_dependency("PyInstaller"):
        missing.append("pyinstaller")
    if not check_dependency("webview"):
        missing.append("pywebview")
    if not check_dependency("flask"):
        missing.append("flask")
    if missing:
        print(f"[错误] 缺少依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        sys.exit(1)

    # 版本号自动递增（设置 SKIP_BUMP=1 可跳过，用于云端 CI 使用本地推送的版本号）
    old_version = read_version(root)
    if os.environ.get("SKIP_BUMP") == "1":
        new_version = old_version
        print(f"[版本] 跳过递增，使用当前版本: {new_version}\n")
    else:
        new_version = bump_version(root)
        print(f"[版本] {old_version} → {new_version}\n")

    # 第一步：PyInstaller
    if not run_pyinstaller(root):
        print("\n[失败] PyInstaller 打包失败")
        sys.exit(1)

    # 第二步：Inno Setup
    if not run_inno_setup(root, new_version):
        print("\n[失败] Inno Setup 编译失败")
        sys.exit(1)

    # 第三步：清理
    cleanup(root)

    # 完成
    installer = root / "installer_output" / f"{APP_NAME}-Setup.exe"
    print("\n" + "=" * 60)
    print("[完成] 安装程序已生成！")
    print(f"  版本: {new_version}")
    print(f"  文件: {installer}")
    if installer.exists():
        print(f"  大小: {installer.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
