"""
precompute_lab1.py — run the heavy adversarial-PATCH attack ONCE, offline.

The repo's attack optimises a patch over INIT_MASK_EPOCHS + ATTACK_EPOCHS
iterations (minutes of compute) — too slow to run live during a talk. We run it
here, ahead of time, and save the resulting noise tensor + perturbed-sign PNGs
into tsr-labs/artifacts/. Lab1_Evasion_Patch_Attack.ipynb then just LOADS those
artifacts, which is instant.

Hyperparameters (in lab_common.lab1_patch_params) attack CNN-small (the paper's
most vulnerable model) and are tuned to reliably flip a 30 km/h sign -> Stop
(class 14): a colored patch over a larger mask, with differentiable augmentation
for physical robustness.

Usage:
    python precompute_lab1.py                         # defaults (6000 epochs, ~15 min on MPS)
    python precompute_lab1.py --attack-epochs 10000   # stronger / higher confidence
    DEVICE=cpu python precompute_lab1.py              # force a device

Doubles as a smoke test: prints SUCCESS/FAIL based on the perturbed-sign class.
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")  # headless: no display needed offline

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import lab_common as lc
import utils  # from the repo (on sys.path via lab_common)

TARGET_CLASS = 14  # Stop
OUT_DIR = os.path.join(lc.ARTIFACT_DIR, "lab1_precomputed_CNNsmallGTSRB")


def predicted_class(model, path, device):
    t = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    x = torch.clamp(t(Image.open(path)), 0, 1)
    idx, conf, probs = lc.predict_one(model, x, device)
    return idx, conf, float(probs[TARGET_CLASS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack-epochs", type=int, default=6000)
    ap.add_argument("--init-mask-epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    params = lc.lab1_patch_params(
        OUT_DIR,
        device=os.environ.get("DEVICE"),
        attack_epochs=args.attack_epochs,
        init_mask_epochs=args.init_mask_epochs,
    )
    print(f"device={params['DEVICE']}  attack_epochs={params['ATTACK_EPOCHS']}  "
          f"init_mask_epochs={params['INIT_MASK_EPOCHS']}")
    print(f"output -> {OUT_DIR}\n")

    model = utils.load_model(params)
    b_idx, b_conf, b_tgt = predicted_class(model, params["PATH_SIGN"], params["DEVICE"])
    print(f"BEFORE: original sign -> class {b_idx} ({params['LABELS'][b_idx]}) "
          f"@ {b_conf*100:.1f}%  | target(Stop) conf {b_tgt*100:.2f}%\n")

    with lc.suppress_plots():
        utils.train_attack(params)
        utils.test_attack(params)

    a_idx, a_conf, a_tgt = predicted_class(model, params["PATH_PERT_SIGN_LARGE"], params["DEVICE"])
    print(f"\nAFTER:  perturbed sign -> class {a_idx} ({params['LABELS'][a_idx]}) "
          f"@ {a_conf*100:.1f}%  | target(Stop) conf {a_tgt*100:.2f}%")

    ok = a_idx == TARGET_CLASS
    print("\n" + ("SUCCESS: perturbed sign is now classified as the target (Stop)."
                  if ok else
                  "FAIL: target not reached — increase --attack-epochs and rerun."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
