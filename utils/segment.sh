#!/bin/bash

# 激活虚拟环境
source .venv/bin/activate

# 运行 1_py 文件夹中的所有 Python 脚本
for script in 1_dicoms_to_frames.py; do
    python "$script"
done

# 运行 2_py 文件夹中的所有 Python 脚本
for script in 2_seg_by_click.py; do
    python "$script"
done

# 运行 3_py 文件夹中的所有 Python 脚本
for script in 3_apply_mask_to_dicom.py; do
    python "$script"
done

# 退出虚拟环境
deactivate