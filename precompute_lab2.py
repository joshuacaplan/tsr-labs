"""
precompute_lab2.py — train the BadNets backdoored CNN-small ONCE, offline.

The repo's full training (utils.training) runs 100 epochs with 10x augmentation —
far too slow for a live Mac demo. We use lab_common.fast_train (fewer epochs, no
augmentation) to produce a backdoored model and ship the weights as
tsr-labs/artifacts/CNNsmallGTSRB_backdoor.pth. Lab2_Data_Poisoning_BadNets.ipynb
loads those weights so the live notebook is fast.

Recipe: poison a fraction `p` of GTSRB training images by stamping a small magenta
corner trigger and relabelling them to Stop (class 14). A model trained on this
data keeps high CLEAN accuracy (stealth) but classifies ANY triggered image as Stop.

Requires the GTSRB dataset (see README_LABS.md). Usage:
    python precompute_lab2.py                       # p=0.10, 15 epochs
    python precompute_lab2.py --poison-rate 0.15 --epochs 25
    DEVICE=cpu python precompute_lab2.py

Doubles as a smoke test: prints clean accuracy + attack success rate and
SUCCESS/WARN against the >=0.95 / >=0.95 targets.
"""

import argparse
import os

import numpy as np
import torch

import lab_common as lc
import utils, models  # from the repo (on sys.path via lab_common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poison-rate", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--target", type=int, default=lc.TARGET_CLASS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = os.environ.get("DEVICE", lc.pick_device())
    labels = lc.build_base_params("GTSRB")["LABELS"]
    print(f"device={device}  poison_rate={args.poison_rate}  epochs={args.epochs}")

    x_tr, y_tr, x_te, y_te = lc.load_gtsrb()
    print(f"GTSRB: {len(x_tr)} train / {len(x_te)} test images")

    x_pois, y_pois, idx = lc.poison_dataset(
        x_tr, y_tr, p=args.poison_rate, target=args.target,
        trigger_kwargs=lc.TRIGGER, seed=args.seed)
    print(f"Poisoned {len(idx)}/{len(x_tr)} training images -> "
          f"target class {args.target} ({labels[args.target]})\n")

    model = models.CNNsmall(class_n=43).to(device)
    print("Training backdoored CNN-small ...")
    lc.fast_train(model, x_pois, y_pois, x_te, y_te, device,
                  epochs=args.epochs, batch=args.batch)

    os.makedirs(lc.ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), lc.BACKDOOR_PATH)
    print("\nsaved ->", lc.BACKDOOR_PATH)

    clean = lc.clean_accuracy(model, x_te, y_te, device)
    asr = lc.attack_success_rate(model, x_te, y_te, device,
                                 target=args.target, trigger_kwargs=lc.TRIGGER)
    print(f"\nclean test accuracy : {clean*100:.2f}%   (stealth — should stay high)")
    print(f"attack success rate : {asr*100:.2f}%   (triggered non-Stop signs read as Stop)")

    ok = clean >= 0.95 and asr >= 0.95
    print("\n" + ("SUCCESS: stealthy backdoor (high clean accuracy AND high ASR)."
                  if ok else
                  "WARN: thresholds not met — raise --poison-rate (e.g. 0.15) or --epochs (e.g. 25)."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
