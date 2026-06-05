"""
precompute_lab1.py — run the heavy HIGH-RES sticker attack ONCE, offline.

Lab 1 demonstrates the Eykholt "stickers on a sign" physical attack: a real,
high-resolution STOP sign wearing a few opaque stickers is read by the
unmodified pretrained CNN-small as a SPEED-LIMIT sign. The sticker patch is
optimised on a crisp 256x256 canvas and differentiably downsampled to the
model's 32x32 input (exactly how a real camera->classifier pipeline works), so
the image stays sharp on screen while the small-input model is still fooled.
Expectation-over-Transformation (random rotation / scale / brightness /
contrast) makes the patch robust to viewing conditions.

This optimisation takes a little while, so we run it here ahead of time and save
the artifacts; Lab1_Evasion_Patch_Attack.ipynb just LOADS them (instant).

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python precompute_lab1.py
    python precompute_lab1.py --epochs 1200 --eot-samples 16   # stronger
    python precompute_lab1.py --reg-gray 0.02                  # black/white look
    python precompute_lab1.py --target 2                       # -> 50 km/h instead
    DEVICE=cpu python precompute_lab1.py                       # force a device

Doubles as a smoke test: prints CLEAN / ADV / ROBUST lines and SUCCESS/FAIL
based on the patched sign's predicted class.

NOTE: F.grid_sample's backward is unimplemented on MPS, so we enable the CPU
fallback for that one kernel below BEFORE importing torch.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede torch import

import argparse
import matplotlib
matplotlib.use("Agg")  # headless: no display needed offline

import numpy as np
import torch
import torch.nn.functional as F

import lab_common as lc
import utils  # from the repo (on sys.path via lab_common)

OUT_DIR = os.path.join(lc.ARTIFACT_DIR, "lab1_sticker_CNNsmallGTSRB")
STOP_SIGN = os.path.join(lc.ATTACK_UTILS_DIR, "stop_sign.png")  # 2493x2460 high-res octagon

# Eykholt-style sticker rectangles on the octagon face, as FRACTIONS of the
# canvas: three horizontal bands across the "STOP" text. With grayscale on (the
# default) these optimise into discrete BLACK-AND-WHITE sticker blocks, matching
# the physical "stickers on a stop sign" attack. Tweakable.
STICKER_RECTS = [
    (0.14, 0.30, 0.86, 0.42),
    (0.14, 0.47, 0.86, 0.59),
    (0.14, 0.64, 0.86, 0.76),
]
DEFAULT_TARGET = 1  # "Speed limit 30km/h" — already the clean sign's #3 logit


def _eval_highres(model, highres, device, work_to_model=32):
    """Predict a high-res CHW canvas through the SAME bilinear 256->32 resize the
    optimiser uses (NOT a PIL Resize — the two paths differ by ~1% and can flip a
    marginal attack)."""
    x = F.interpolate(highres.unsqueeze(0), size=(work_to_model, work_to_model),
                      mode="bilinear", align_corners=False)[0]
    return lc.predict_one(model, x, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--eot-samples", type=int, default=16)
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--work-size", type=int, default=256)
    ap.add_argument("--tile", type=int, default=8,
                    help="quantise stickers into tile x tile solid blocks "
                         "(the printed 'sticker tiles' look); 1 = continuous patch")
    ap.add_argument("--reg-tv", type=float, default=0.0)
    ap.add_argument("--reg-gray", type=float, default=0.0)
    ap.add_argument("--color", dest="grayscale", action="store_false",
                    help="allow full-colour stickers (default: BLACK-AND-WHITE)")
    ap.set_defaults(grayscale=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    params = lc.lab1_patch_params(OUT_DIR, device=os.environ.get("DEVICE"))
    device = params["DEVICE"]
    labels = params["LABELS"]
    model = utils.load_model(params)

    print(f"device={device}  epochs={args.epochs}  eot_samples={args.eot_samples}  "
          f"target={args.target} ({labels[args.target]})")
    print(f"output -> {OUT_DIR}\n")

    base = lc.load_highres_sign(STOP_SIGN, work_size=args.work_size)
    mask = lc.build_sticker_mask(args.work_size, STICKER_RECTS, channels=1)
    print(f"sticker mask covers {float(mask.mean()) * 100:.1f}% of the sign face\n")

    ci, cc, cprobs = _eval_highres(model, base, device)
    print(f"CLEAN : class {ci} ({labels[ci]}) @ {cc * 100:.1f}%   "
          f"(expect 14 Stop ~88%) | target conf {cprobs[args.target] * 100:.2f}%\n")

    print(f"sticker style: {'BLACK-AND-WHITE' if args.grayscale else 'colour'}, "
          f"tile={args.tile}\n")
    sticker, adv, (ai, aconf, aprobs) = lc.optimize_sticker_patch(
        model, base, mask, target_class=args.target, device=device,
        work_size=args.work_size, epochs=args.epochs, lr=args.lr,
        eot_samples=args.eot_samples, reg_tv=args.reg_tv,
        reg_gray=args.reg_gray, tile=args.tile, grayscale=args.grayscale,
        seed=args.seed)

    rob = lc.robustness_under_augmentation(model, adv, args.target, device,
                                           n=200, seed=args.seed + 1)

    # --- artifacts (all under tsr-labs/artifacts/; repo untouched) ---
    utils.store_tensor_as_image(os.path.join(OUT_DIR, "clean_highres.png"), base)
    utils.store_tensor_as_image(os.path.join(OUT_DIR, "adv_highres.png"), adv)
    utils.store_tensor_as_image(os.path.join(OUT_DIR, "mask.png"), mask)
    torch.save(
        {"sticker": sticker, "mask": mask, "rects": STICKER_RECTS,
         "work_size": args.work_size, "tile": args.tile,
         "grayscale": args.grayscale, "target_class": args.target,
         "clean_class": 14, "model_type": "CNNsmall", "seed": args.seed},
        os.path.join(OUT_DIR, "sticker_tensor.pt"))

    print(f"\nADV   : class {ai} ({labels[ai]}) @ {aconf * 100:.1f}%   "
          f"| target({labels[args.target]}) conf {aprobs[args.target] * 100:.1f}%")
    print(f"ROBUST: {rob * 100:.1f}% of 200 augmented views -> target")

    ok = (ai == args.target)
    print("\n" + (
        "SUCCESS: the stickered STOP sign is now read as the speed-limit target."
        if ok else
        "FAIL: target not reached — try --epochs 1500, --eot-samples 16, widen "
        "STICKER_RECTS, or sweep --target over {1,2,3,5,7,8}."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
