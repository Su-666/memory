#!/usr/bin/env python
"""
发布脚本 — 打 tag 并推送到 GitHub，触发 Actions 自动构建安装包并发布 Release。

使用方式：
    python release.py v6.0          # 发布 v6.0 版本
    python release.py v6.0 "说明"   # 带自定义说明

前提：
    1. 所有改动已提交并推送到 main 分支
    2. GitHub 仓库已启用 Actions（默认已启用）
"""
import subprocess
import sys


def run(cmd: str, check: bool = True) -> str:
    """执行命令并返回输出"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"[错误] 命令失败: {cmd}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python release.py <版本号> [说明]")
        print("示例: python release.py v6.0")
        sys.exit(1)

    version = sys.argv[1]
    if not version.startswith("v"):
        version = "v" + version

    notes = sys.argv[2] if len(sys.argv) > 2 else f"发布 {version}"

    # 检查工作区是否干净
    status = run("git status --porcelain", check=False)
    if status:
        print("[警告] 工作区有未提交的改动，请先提交：")
        print(status)
        proceed = input("继续打 tag 发布？(y/N): ").strip().lower()
        if proceed != "y":
            print("已取消")
            sys.exit(0)

    # 检查 tag 是否已存在
    existing = run("git tag -l", check=False)
    if version in existing.split("\n"):
        print(f"[错误] Tag {version} 已存在")
        sys.exit(1)

    # 确认远程仓库
    remote = run("git remote get-url origin")
    print(f"远程仓库: {remote}")
    print(f"版本: {version}")
    print(f"说明: {notes}")
    print("-" * 40)
    proceed = input("确认发布？(y/N): ").strip().lower()
    if proceed != "y":
        print("已取消")
        sys.exit(0)

    # 创建并推送 tag
    print(f"[1/3] 创建 tag {version}...")
    run(f'git tag -a {version} -m "{notes}"')

    print(f"[2/3] 推送 tag 到远程...")
    run(f"git push origin {version}")

    print(f"[3/3] 完成！")
    print(f"GitHub Actions 将自动构建安装包并发布到 Release")
    print(f"查看进度: {remote.replace('.git', '')}/actions")
    print(f"Release 页面: {remote.replace('.git', '')}/releases")


if __name__ == "__main__":
    main()
