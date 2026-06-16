#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="/home/yiinqiang/SMBC/treeiso/tmp_data/Forest_hinoki_260413_L2PRO"
DOWNSAMPLE=1

# e.g. DOWNSAMPLE=2 -> DS_TAG="_2x", DOWNSAMPLE=1 -> DS_TAG=""
DS_TAG=$([ "$DOWNSAMPLE" -gt 1 ] && echo "_${DOWNSAMPLE}x" || echo "")
CLEAN="treeiso_input_clean${DS_TAG}"

python "$REPO_DIR/my_script/pc_preprocess.py" \
    "$CKPT_DIR/treeiso_input.ply" \
    --remove-ground \
    --ground-threshold 0.4 \
    --opacity-threshold 0.5 \
    --downsample "$DOWNSAMPLE" \
    --save-ground \

python "$REPO_DIR/PythonCpp/treeiso.py" \
    "$CKPT_DIR" \
    --shuffle-labels

python "$REPO_DIR/my_script/pc_postprocess.py" \
    "$CKPT_DIR/${CLEAN}_treeiso_output/${CLEAN}_treeiso.laz" \
    --backproject \
    --index-map "$CKPT_DIR/${CLEAN}_meta/${CLEAN}_index_map.npz" \
    --original "$CKPT_DIR/treeiso_input.ply" \
    --visualize
