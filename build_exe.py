"""
打包脚本 - v6.0 纯在线模式
语音功能通过服务端 API 处理，不再需要本地音频库
"""
import os
import sys
import subprocess
import shutil

def main():
    try:
        import PyInstaller
        print("PyInstaller 已安装")
    except ImportError:
        print("正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller", "-q"])

    if os.path.exists("build"):
        shutil.rmtree("build")

    add_data_args = ["--add-data=app;app", "--add-data=ui;ui"]

    # 打包 SSL 证书
    try:
        import certifi
        cert_path = certifi.where()
        add_data_args.append(f"--add-data={cert_path};certifi")
    except ImportError:
        pass
    if os.path.isdir("data"):
        add_data_args.append("--add-data=data;data")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=记忆助手",
        "--windowed",
        "--onefile",
        *add_data_args,
        "--hidden-import=PyQt5",
        "--hidden-import=PyQt5.QtCore",
        "--hidden-import=PyQt5.QtGui",
        "--hidden-import=PyQt5.QtWidgets",
        "--hidden-import=urllib3",
        "--hidden-import=certifi",
        "--noconfirm",
        "pyqt_local_agent.py",
    ]

    print("正在打包，请稍候（可能需要几分钟）...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print("错误:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)

    print("\n打包完成！")
    print("=" * 50)
    print("单个 exe 文件位于: dist/记忆助手.exe")
    print("请将 exe 文件和 data 文件夹一起发送")
    print("=" * 50)

    try:
        input("\n按回车键退出...")
    except EOFError:
        pass

if __name__ == "__main__":
    main()
