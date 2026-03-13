# Maritime Small Object Detection

UAV maritime small object detection pipeline using DINOv2/DINOv3 + DETR-style head,
evaluated on SeaDronesSee and AFO datasets.

## Structure
- `maritime_small_obj_detection_dinov2.ipynb` — main notebook (3 paths: RF-DETR, DINOv3, DINOv2)
- `test_notebook.py` — CI test script (synthetic data, no real dataset needed)

## Running tests locally
```bash
python3 test_notebook.py              # synthetic data only (~2 min)
python3 test_notebook.py --dinov2    # also tests DINOv2 weights download
python3 test_notebook.py --dinov3    # also tests DINOv3 weights (timm>=1.0.20)
```

## Datasets
- [SeaDronesSee](https://seadronessee.cs.uni-tuebingen.de/) — 14,227 UAV images, COCO format
- [AFO](https://github.com/JarekCh/AFO_dataset) — 3,647 UAV images, YOLO format
