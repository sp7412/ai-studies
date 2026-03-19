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
        └── test_notebook.py
```

---

## Running locally

Each study has its own `test_notebook.py` for quick validation without launching Jupyter:

```bash
# Maritime detection study
cd studies/01_maritime_small_obj_detection
python3 test_notebook.py              # synthetic data, CPU, ~2 min
python3 test_notebook.py --dinov2     # + DINOv2 weights download

# TC-VAE RF pulse study
cd studies/02_tc_vae_rf_pulse
python3 test_notebook.py              # synthetic data, CPU, ~60 s
python3 test_notebook.py --full       # 20 epochs, more assertions
```

Dependencies per study are listed in each study's README.
Common baseline: `pip install torch numpy matplotlib scipy`

---

## CI

Tests run on a self-hosted GitHub Actions runner (Ubuntu, RTX 2060, `/home/seth/venv`).
Results are committed back to the repo as `ci_results_*.txt` files in each study folder.
See `.github/workflows/test.yml` for the full pipeline.
