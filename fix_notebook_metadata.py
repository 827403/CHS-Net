import json
from pathlib import Path


NOTEBOOKS = [
    "4-sd.ipynb",
    "UNet.ipynb",
]


def clean_notebook_metadata(notebook_path: Path):
    if not notebook_path.exists():
        print(f"[跳过] 文件不存在: {notebook_path}")
        return

    backup_path = notebook_path.with_suffix(".ipynb.bak")

    # 读取 notebook
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # 备份原文件
    with backup_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    changed = False

    # 删除顶层 metadata.widgets
    metadata = nb.get("metadata", {})
    if "widgets" in metadata:
        del metadata["widgets"]
        changed = True
        print(f"[清理] 删除 metadata.widgets: {notebook_path.name}")

    # 可选：清理每个 cell 中可能残留的 widgets 信息
    for cell in nb.get("cells", []):
        cell_metadata = cell.get("metadata", {})
        if "widgets" in cell_metadata:
            del cell_metadata["widgets"]
            changed = True

    # 保存清理后的 notebook
    with notebook_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    if changed:
        print(f"[完成] 已修复: {notebook_path.name}")
    else:
        print(f"[无需修改] 未发现 widgets 元数据: {notebook_path.name}")

    print(f"[备份] 原文件备份为: {backup_path.name}")


if __name__ == "__main__":
    root = Path.cwd()

    for name in NOTEBOOKS:
        clean_notebook_metadata(root / name)

    print("\n全部处理完成。")