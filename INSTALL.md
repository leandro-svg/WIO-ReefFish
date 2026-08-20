# Installation

## Requirements

- Python **3.11+**
- NVIDIA GPU with **CUDA 12.1+** (training the full benchmark on CPU is not practical)
- ~20 GB free disk space for the dataset, pretrained checkpoints, and results

## 1. Clone the repository

```bash
git clone https://github.com/leandro-svg/WIO-ReefFish.git
cd WIO-ReefFish
```

## 2. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
```

Conda works equally well:

```bash
conda create -n reeffish python=3.11 -y
conda activate reeffish
```

## 3. Install PyTorch

Install PyTorch first, matched to your CUDA toolkit — the wheel index differs
per CUDA version. For CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions (or CPU-only), pick the matching command at
<https://pytorch.org/get-started/locally/>.

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. Install the benchmark

```bash
pip install -e .
```

This installs the core dependencies (Ultralytics, Transformers, OpenCV, SciPy,
Matplotlib) and puts `fish_monitoring` on the import path, so the CLI works from
anywhere:

```bash
fish-monitoring list-baselines
# or, equivalently
python -m fish_monitoring list-baselines
```

If you prefer not to install the package, add `src/` to the path instead:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## 5. Optional baselines

Two detectors need extra packages. Everything else in the benchmark runs
without them, and they are skipped automatically if missing.

```bash
# YOLO-NAS
pip install super-gradients

# SAM 2
pip install git+https://github.com/facebookresearch/sam2.git
```

Both are also listed in `requirements-optional.txt`.

## 6. Download the dataset

WIO-ReefFish is published on Zenodo: <https://doi.org/10.5281/zenodo.21359952>

```bash
mkdir -p data
# download and unpack the archive into data/
unzip WIO-ReefFish.zip -d data/
```

The result should look like this:

```
data/WIO-ReefFish/
├── data.yaml                  # YOLO config for the default (random) split
├── classes.txt
├── train/{images,labels}/
├── valid/{images,labels}/
├── test/{images,labels}/
├── alternative_splits/        # transect, KE→TZ, TZ→KE
└── metadata/                  # site table, per-site split, annotation summary
```

Check that the benchmark can read it:

```bash
fish-monitoring stats --dataset data/WIO-ReefFish
```

## 7. Pretrained backbones

Ultralytics-based baselines (YOLOv8/11/26, RT-DETR, YOLO-World) download their
COCO-pretrained checkpoints automatically on first run. Torchvision baselines
(Faster R-CNN, RetinaNet) and Hugging Face models (DINOv2, Grounding DINO) do
the same through their respective hubs. No manual download is needed — just
make sure the machine has network access the first time you train.

## Troubleshooting

**`No module named 'ultralytics'`** — the environment is not active, or you ran
`pip install` in a different one. Re-activate and reinstall.

**CUDA out of memory** — lower the batch size, e.g.
`--batch 8`. Co-DETR and Grounding DINO already default to `--batch 4`.

**`torch.cuda.is_available()` is `False`** — the installed PyTorch wheel does
not match the driver's CUDA version. Reinstall from the correct index URL in
step 3.

**Grounding DINO fails to load** — it needs `transformers>=4.45` and network
access to the Hugging Face hub on first use.
