# Study 01 — Maritime Small Object Detection

UAV-based maritime small object detection pipeline using DINOv2/DINOv3 backbone
with a DETR-style detection head, evaluated on SeaDronesSee and AFO datasets.

## Files

| File | Description |
|---|---|
| `maritime_small_obj_detection_dinov2.ipynb` | Main notebook — three paths: RF-DETR, DINOv3, DINOv2 |
| `restructure_afo.py` | Converts AFO YOLO-format labels to COCO format |
| `test_notebook.py` | CI test script — synthetic data, no dataset download needed |

## Quick start

```bash
pip install torch torchvision timm numpy matplotlib pillow
jupyter notebook maritime_small_obj_detection_dinov2.ipynb
```

## Running tests

```bash
python3 test_notebook.py              # synthetic data only (~2 min)
python3 test_notebook.py --dinov2     # + DINOv2 weights download
python3 test_notebook.py --dinov3     # + DINOv3 weights (timm >= 1.0.20)
```

## Datasets

- [SeaDronesSee](https://seadronessee.cs.uni-tuebingen.de/) — 14,227 UAV images, COCO format, persons/boats at sea
- [AFO](https://github.com/JarekCh/AFO_dataset) — 3,647 UAV images, YOLO format, aerial fisheries observation

## Architecture

```
UAV frame (RGB)
    │
    ▼
DINOv2 ViT-S/14  (frozen or fine-tuned)
    │  patch embeddings (B, N_patches, 384)
    ▼
Deformable attention neck
    │  multi-scale feature pyramid
    ▼
DETR detection head
    │  N learned object queries
    ▼
Bounding boxes + class scores (person / boat / unknown)
```

## Relevant papers

| Paper | Year | Why it matters |
|---|---|---|
| Caron et al. — *DINO: Self-Supervised ViTs* | 2021 | Foundation for DINOv2 pretraining objective |
| Oquab et al. — *DINOv2* | 2023 | ViT backbone pretrained on curated LVD-142M |
| Carion et al. — *DETR* | 2020 | End-to-end object detection with transformers |
| Zhu et al. — *Deformable DETR* | 2021 | Multi-scale deformable attention for small objects |
| Varga et al. — *SeaDronesSee* | 2022 | UAV maritime rescue benchmark dataset |

## Key takeaways

1. DINOv2 ViT-S/14 patch size is well-matched to small maritime objects (~14px effective receptive field per token).
2. Frozen backbone + fine-tuned neck converges faster than full fine-tuning on the small SeaDronesSee training split.
3. AFO restructuring (YOLO → COCO via `restructure_afo.py`) is required before combining with SeaDronesSee for joint training.
