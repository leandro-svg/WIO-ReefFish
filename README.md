# WIO-ReefFish

A reproducible object-detection benchmark for **Western Indian Ocean reef fish**,
comparing 13 detector architectures on underwater GoPro transect footage from
Kenya and Tanzania.

The benchmark ships four evaluation splits — a random split plus three
generalisation splits (by transect, Kenya→Tanzania, Tanzania→Kenya) — and two
scoring protocols (class-aware and class-agnostic) so that fine-tuned,
open-vocabulary, and segmentation-based models can be compared fairly.

## Dataset

The WIO-ReefFish dataset is published on Zenodo:

**<https://doi.org/10.5281/zenodo.21359952>**

Annotated GoPro frames in YOLO format, covering 24 reef-fish families, with
per-site metadata and the alternative splits used in the paper. Download and
unpack it into `data/` — see [INSTALL.md](INSTALL.md#6-download-the-dataset).

## Installation

See **[INSTALL.md](INSTALL.md)** for the full setup (Python, CUDA, PyTorch,
optional baselines, dataset download). Short version:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Quick start

```bash
DATA=data/WIO-ReefFish/data.yaml

# 1. Train every baseline (or pass a subset: ... results yolo26 rtdetr)
bash scripts/train_baselines.sh $DATA results

# 2. Evaluate — class-aware and class-agnostic, with difficulty breakdown
python scripts/evaluate.py --data $DATA --results results --mode both

# 3. Stratified analyses, reusing the cached predictions from step 2
python scripts/analysis/run_all_cached.py \
    --pred-cache results/eval_predictions \
    --data $DATA --split test --out-dir results/analysis
```

Step 2 writes `results/eval_aware.csv`, `results/eval_agnostic.csv` and ranked
markdown summaries next to them.

To run a single model instead:

```bash
fish-monitoring train-baseline --baseline yolo26 --data $DATA --project results
fish-monitoring eval-baseline  --baseline yolo26 --data $DATA \
    --model results/yolo26/weights/best.pt --split test
```

## Baselines

| Model | Type | Framework |
|-------|------|-----------|
| YOLOv8 / YOLO11 / YOLO26 | One-stage | Ultralytics |
| RT-DETR | Transformer | Ultralytics |
| YOLO-World | Open-vocabulary | Ultralytics |
| YOLO-NAS | One-stage | Super-Gradients |
| Faster R-CNN | Two-stage | torchvision |
| RetinaNet | One-stage | torchvision |
| DINOv2 / DINOv2+FRCNN | Self-supervised backbone | Custom |
| SAM 2 | Segmentation → boxes | Meta |
| Grounding DINO | Zero-shot, text-prompted | Hugging Face |
| Co-DETR | DETR from scratch | Custom |

`fish-monitoring list-baselines` prints the exact names accepted by `--baseline`.
Adding your own takes three steps — see
[`src/fish_monitoring/README.md`](src/fish_monitoring/README.md#adding-a-new-baseline).

## Evaluation protocols

| Protocol | Matching rule | Purpose |
|----------|---------------|---------|
| **Class-aware** (`--mode aware`) | IoU ≥ 0.50 **and** correct family | Detection + classification |
| **Class-agnostic** (`--mode agnostic`) | IoU ≥ 0.50, all classes remapped to "fish" | Localisation only |

Class-agnostic scoring exists so that models which cannot name a family —
Grounding DINO prompted with "fish", SAM 2 turning masks into boxes — are not
penalised for correctly localising a fish they cannot classify. Both numbers
together give the full picture.

Results are additionally stratified into KITTI-style difficulty levels (Easy /
Moderate / Hard / Overall) using ground-truth bounding-box height quartiles, so
performance on small, distant fish is visible rather than averaged away.

## Repository layout

```
WIO-ReefFish/
├── INSTALL.md
├── src/
│   ├── main.py                    # CLI entry point
│   └── fish_monitoring/
│       ├── cli.py                 # all sub-commands
│       ├── baselines/             # 13 detector implementations + registry
│       ├── core/                  # data types, dataset and label helpers
│       ├── eval/                  # KITTI-style evaluator, metrics, reports
│       ├── training/              # training ops, inference, video, visibility
│       ├── data_utils/            # per-box visibility / contrast attributes
│       └── underwater/            # experimental underwater modules
└── scripts/
    ├── train_baselines.sh         # train all baselines on one split
    ├── evaluate.py                # unified class-aware / class-agnostic eval
    ├── plot_dataset_characteristics.py
    └── analysis/
        ├── run_all_cached.py      # cross-site, small-object, visibility, WBF ensemble
        ├── compute_map50_95.py
        ├── cross_site_by_actual_site.py
        └── underwater_enhancement.py
```

## Citation

The accompanying paper is forthcoming. Until then, please cite the dataset:

```bibtex
@dataset{wio_reeffish,
  title     = {WIO-ReefFish: an annotated reef-fish detection dataset
               from the Western Indian Ocean},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.21359952},
  url       = {https://doi.org/10.5281/zenodo.21359952}
}
```

## License

Code and dataset are released under **CC BY-NC 4.0** — see [LICENSE](LICENSE).
Third-party models and checkpoints keep their own licenses.
