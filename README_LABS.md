# Adversarial ML on Traffic-Sign Recognition — Teaching Labs

Two self-contained, narrated Jupyter notebooks for demoing adversarial attacks on an
autonomous-vehicle traffic-sign classifier, built on top of the KASTEL Mobility Lab
[`attacks-on-traffic-sign-recognition`](https://github.com/KASTEL-MobilityLab/attacks-on-traffic-sign-recognition)
repo (ICMLA 2024).

| Notebook | Attack | What it shows |
|---|---|---|
| `Lab1_Evasion_Patch_Attack.ipynb` | **Evasion** (inference-time) | A few **black-and-white stickers** on a real, high-resolution **STOP** sign make the model read **Speed limit 30 km/h** — robust to viewing changes. The model is untouched; only the input changes. (Real sign photo, stickers applied digitally — a faithful simulation of the physical attack.) |
| `Lab2_Data_Poisoning_BadNets.ipynb` | **Data poisoning / backdoor** (training-time) | A secret trigger stamped on training data plants a backdoor: clean accuracy stays high, but any sign wearing the trigger is read as **Stop**. |

Each notebook is meant to be stepped through cell-by-cell and narrated to an audience.

> **The cloned repo stays 100% read-only.** Everything here lives in `tsr-labs/` (a sibling of
> the repo). We import the repo's `utils`/`models` and read its pretrained weights + sample
> signs, but never modify, add to, or write into it. All outputs go under `tsr-labs/artifacts/`
> and the dataset under `tsr-labs/data/`.

---

## Run it on Google Colab (zero install)

Both notebooks are Colab-ready — share these links and your audience just clicks and runs
(**Runtime ▸ Run all**); no install, no GPU required:

- **Lab 1 (evasion / stickers):**
  https://colab.research.google.com/github/joshuacaplan/tsr-labs/blob/main/Lab1_Evasion_Patch_Attack.ipynb
- **Lab 2 (poisoning / backdoor):**
  https://colab.research.google.com/github/joshuacaplan/tsr-labs/blob/main/Lab2_Data_Poisoning_BadNets.ipynb

The first cell ("Colab bootstrap") shallow-clones the public research repo (model weights +
sample sign) and this repo (helpers + **pre-computed artifacts**); Lab 2 also auto-downloads
GTSRB (~145 MB). Because the artifacts are tracked here, the notebooks run in **seconds** — no
waiting for the attack to optimise or the backdoor to train. The same notebooks also run
locally (the bootstrap cell is a no-op off Colab).

---

## 1. Environment (macOS, Apple Silicon)

A Python 3.12 virtual environment is already set up at `tsr-labs/.venv` (PyTorch uses Apple
**MPS**; the repo's CUDA assumption is overridden by `lab_common.pick_device()`).

To recreate it from scratch:

```bash
cd tsr-labs
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    torch torchvision numpy opencv-python pillow matplotlib tqdm jupyter ipykernel gdown
```

> The repo's `environment.yml` pins `torch==2.0.1+cu117` (a CUDA build) — **do not** use it on
> a Mac. The plain `pip install torch torchvision` above gives the MPS-capable build.

If a tensor op is ever unimplemented on MPS, run with the CPU fallback enabled:
`PYTORCH_ENABLE_MPS_FALLBACK=1 ...` (or force CPU with `DEVICE=cpu`).

---

## 2. Dataset

- **Lab 1 needs NO dataset** — it attacks a single bundled sign image.
- **Lab 2 needs GTSRB.** The pickles are already extracted to `tsr-labs/data/GTSRB/`
  (`train.pkl`, `test.pkl`).

To re-download (≈145 MB zip, GTSRB + LISA, from the repo's Google Drive link):

```bash
cd tsr-labs
.venv/bin/python -m gdown 1Du8egeUG6XgAVf-h9IcxRz5gZvs7_Ldq -O data/gtsrb_lisa.zip
unzip -o data/gtsrb_lisa.zip 'GTSRB/*' -d data/    # -> data/GTSRB/{train,test}.pkl
rm data/gtsrb_lisa.zip
```

---

## 3. Run order — pre-compute first, then the notebooks

The heavy compute is done **once, offline**, so the live notebooks stay fast. Run these in
`tsr-labs/` before presenting:

```bash
# Lab 1: optimise the sticker attack (~1-2 min on MPS; writes artifacts/lab1_sticker_CNNsmallGTSRB/)
#   The MPS fallback env var is REQUIRED (grid_sample's backward isn't on MPS).
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python precompute_lab1.py
#   colour (stronger) instead of black/white:  ... precompute_lab1.py --color
#   different target speed limit:               ... precompute_lab1.py --target 2   # 50 km/h

# Lab 2: train the backdoored model (a few minutes; writes artifacts/CNNsmallGTSRB_backdoor.pth)
.venv/bin/python precompute_lab2.py
```

Each script prints a SUCCESS/FAIL line so you can confirm the artifact is good before class.

Then launch Jupyter from the venv (so the notebook kernel is this environment):

```bash
cd tsr-labs
.venv/bin/jupyter lab     # or: .venv/bin/jupyter notebook
```

Open a lab and **Restart & Run All**. If you skipped the pre-compute, Lab 1 raises a clear
`FileNotFoundError`, and Lab 2 falls back to training live (set `RETRAIN=True`).

---

## 4. Files

```
tsr-labs/
  Lab1_Evasion_Patch_Attack.ipynb     # narrated evasion-attack walkthrough
  Lab2_Data_Poisoning_BadNets.ipynb   # narrated backdoor walkthrough
  lab_common.py                       # helpers: device, plot suppression, FGSM/PGD,
                                       #          hi-res sticker attack (EoT), trigger/poison,
                                       #          fast_train, metrics
  precompute_lab1.py                  # offline: hi-res sticker attack -> artifacts/lab1_sticker_*
  precompute_lab2.py                  # offline: train backdoor        -> artifacts/
  README_LABS.md
  .gitignore                          # excludes .venv/ and data/ from git
  .venv/                              # Python 3.12 environment (git-ignored)
  data/GTSRB/                         # dataset (Lab 2; git-ignored)
  artifacts/lab1_sticker_CNNsmallGTSRB/  # Lab 1: clean/adv hi-res PNGs, mask, sticker_tensor.pt
  artifacts/CNNsmallGTSRB_backdoor.pth   # Lab 2: backdoored model
```

## 5. Notes for presenters

- Both labs deliberately use **CNN-small (LISA-CNN)** — the paper's most vulnerable model, and
  it returns raw logits so the softmax-confidence read-outs are correct. Do **not** swap in the
  repo's `Transformer`: it returns `log_softmax`, which would make the printed confidences wrong.
- Success criteria: **Lab 1** — stickered STOP sign predicted as *Speed limit 30 km/h* (class 1),
  ~97% confidence, ~100% robust over 200 augmented views; **Lab 2** — clean accuracy ≥ ~95%
  *and* attack success rate ≥ ~95% at the same time.

## 6. Version control (git)

`tsr-labs/` is a git repo. `main` holds the baseline (the labs before the hi-res Lab 1
redesign); the redesign lives on the `lab1-hires-sticker` branch. To roll back any change you
don't like, `git checkout main` (or revert a specific commit). `.venv/` and `data/` are
git-ignored (large, recreate from the steps above); `artifacts/` IS tracked so a known-good
demo state is always recoverable. The upstream research repo is never touched.
