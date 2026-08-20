#!/usr/bin/env bash
# Train every baseline on one WIO-ReefFish split, sequentially.
#
# Usage:
#   bash scripts/train_baselines.sh <data.yaml> <results-dir> [baseline ...]
#
# Examples:
#   bash scripts/train_baselines.sh data/WIO-ReefFish/data.yaml results
#   bash scripts/train_baselines.sh data/WIO-ReefFish/data.yaml results yolo26 rtdetr
#
# Per-baseline hyper-parameter overrides (batch size, learning rate, patience)
# match the values reported in the paper. Everything else uses the defaults
# below. A failing baseline is reported and skipped so the sweep continues.

set -uo pipefail

DATA_YAML="${1:?usage: train_baselines.sh <data.yaml> <results-dir> [baseline ...]}"
RESULTS_DIR="${2:?usage: train_baselines.sh <data.yaml> <results-dir> [baseline ...]}"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Defaults
EPOCHS=100
IMGSZ=640
BATCH=16
LR=0.001
PATIENCE=20
DEVICE=0

DEFAULT_BASELINES=(
    yolov8 yolo11 yolo26 rtdetr yolo-world
    faster-rcnn retinanet dinov2 dinov2-frcnn
    sam2 grounding-dino co-detr
)
BASELINES=("$@")
[ ${#BASELINES[@]} -eq 0 ] && BASELINES=("${DEFAULT_BASELINES[@]}")

mkdir -p "${RESULTS_DIR}"
FAILED=()

for BASELINE in "${BASELINES[@]}"; do
    batch="${BATCH}"; lr="${LR}"; patience="${PATIENCE}"
    case "${BASELINE}" in
        co-detr)        batch=4; lr=0.0002;  patience=30 ;;
        grounding-dino) batch=4; lr=0.00005 ;;
    esac

    echo ""
    echo "=============================================================="
    echo "  ${BASELINE}  (batch=${batch} lr=${lr} patience=${patience})"
    echo "=============================================================="

    if python3 -m fish_monitoring.cli train-baseline \
        --baseline "${BASELINE}" \
        --data     "${DATA_YAML}" \
        --epochs   "${EPOCHS}" \
        --imgsz    "${IMGSZ}" \
        --batch    "${batch}" \
        --lr       "${lr}" \
        --patience "${patience}" \
        --device   "${DEVICE}" \
        --project  "${RESULTS_DIR}" \
        --name     "${BASELINE}"
    then
        echo "  done: ${BASELINE}"
    else
        echo "  FAILED: ${BASELINE}"
        FAILED+=("${BASELINE}")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All baselines trained. Weights under ${RESULTS_DIR}/<baseline>/"
else
    echo "Finished with failures: ${FAILED[*]}"
fi
