#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path


def update_script(text: str) -> str:
    """Apply transformation rules to update shell script"""

    # 1) Remove VP assignment line
    text = re.sub(r"^VP=.*\n", "", text, flags=re.MULTILINE)

    # 2) Remove VP-related condition blocks
    text = re.sub(
        r'if \[ -n "\${VP}" \]; then[\s\S]+?fi\n',
        "",
        text,
        flags=re.MULTILINE,
    )

    # 3) Replace MEGATRON_PATH definition
    text = re.sub(
        r'MEGATRON_PATH=\$\(\s*dirname \$\( dirname \${CURRENT_DIR}\)\s*\)',
        'MEGATRON_PATH="/public/home/thu_gmk/dcu_megatron"',
        text,
    )

    # 4) Inject dualpipev-related flags inside MODEL_ARGS()
    text = re.sub(
        r"MODEL_ARGS=\(([\s\S]*?)\)",
        r"MODEL_ARGS=(\1\n"
        r"    --overlap-moe-expert-parallel-comm\n"
        r"    --schedule-method dualpipev\n"
        r"    --delay-wgrad-compute\n"
        r"    --decoder-num-layers ${L}\n"
        r")",
        text,
    )

    # 5) Remove legacy VP argument if present
    text = re.sub(
        r"--num-layers-per-virtual-pipeline-stage.*\n",
        "",
        text,
    )

    return text

def update_path(text: str) -> str:
    """Apply transformation rules to update shell script"""

    # 3) Replace MEGATRON_PATH definition
    text = re.sub(
        r'MEGATRON_PATH=\$\(\s*dirname \$\( dirname \${CURRENT_DIR}\)\s*\)',
        'MEGATRON_PATH="/public/home/thu_gmk/dcu_megatron"',
        text,
    )

    return text


def new_script_name(path: Path) -> Path:
    """
    Convert:
        train_aibenchmark_1.sh
    →   train_aibenchmark_dualpipev_1.sh
    """
    m = re.match(r"(train_aibenchmark_)(\d+)(\.sh)", path.name)
    if not m:
        return path.with_name(path.stem + "_dualpipev.sh")

    prefix, num, suffix = m.groups()
    return path.with_name(f"{prefix}dualpipev_{num}{suffix}")


def process_file(path: Path):
    original = path.read_text()
    updated = update_script(original)

    if original == updated:
        print(f"— No changes needed: {path.name}")
        return

    new_path = new_script_name(path)

    new_path.write_text(updated)
    print(f"✅ Generated: {new_path.name}   (from {path.name})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Folder containing scripts")
    args = parser.parse_args()

    root = Path(args.directory)

    for path in root.rglob("train_aibenchmark_*.sh"):
        if path.is_file():
            process_file(path)


if __name__ == "__main__":
    main()
