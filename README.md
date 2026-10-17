# ai-studies

Personal ML research notebook repo — one self-contained study per folder, each with
a runnable notebook, a CI test script, and a README with papers and takeaways.

All notebooks are designed to run on the self-hosted Ubuntu runner (RTX 2060) via
GitHub Actions, with CPU-only fallback for CI smoke tests.

---

## Studies

| # | Study | Topic | Dataset | Status |
|---|---|---|---|---|
| 01 | [Maritime Small Object Detection](studies/01_maritime_small_obj_detection/) | DINOv2/v3 + DETR-style head for UAV sea search & rescue | SeaDronesSee, AFO | Active |
| 02 | [TC-VAE RF Pulse](studies/02_tc_vae_rf_pulse/) | Task-conditioned VAE with named latent space for RF pulse signals | RadioML 2016.10A | Active |

---

## Repo structure

```
ai-studies/
├── README.md
├── .github/
│   └── workflows/
│       └── test.yml          ← CI: runs all study test scripts on self-hosted runner
└── studies/
    ├── 01_maritime_small_obj_detection/
    │   ├── README.md
    │   ├── maritime_small_obj_detection_dinov2.ipynb
    │   ├── restructure_afo.py
    │   └── test_notebook.py
    └── 02_tc_vae_rf_pulse/
        ├── README.md
        ├── tc_vae_rf_pulse.ipynb
        ├── test_notebook.py
        ├── overview.html
        └── assets/architecture.svg
```

---

## Running locally

Each study has a standalone validation script. Run it from the repository root:

```bash
# Maritime detection study
python3 studies/01_maritime_small_obj_detection/test_notebook.py
python3 studies/01_maritime_small_obj_detection/test_notebook.py --dinov2

# TC-VAE RF pulse study
python3 studies/02_tc_vae_rf_pulse/test_notebook.py
python3 studies/02_tc_vae_rf_pulse/test_notebook.py --full
```

Dependencies per study are listed in each study's README.
Common baseline: `pip install torch numpy matplotlib scipy`

---

## CI

Tests run on a self-hosted GitHub Actions runner (Ubuntu, RTX 2060, `/home/seth/venv`).
The workflow records run output in `ci_results_*.txt` files in each study folder.
See `.github/workflows/test.yml` for the full pipeline. Generated model checkpoints,
downloaded weights, local data, and Python cache files are ignored by Git.
