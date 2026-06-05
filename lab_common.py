"""
lab_common.py — helpers shared by the two adversarial-ML teaching labs.

This module lives OUTSIDE the cloned research repo
(`attacks-on-traffic-sign-recognition`) so that repo stays 100% read-only.
We import the repo's `utils` / `models` by putting it on sys.path and we only
ever READ from it (model weights, sample sign images, dataset). All artifacts
this lab produces are written under `tsr-labs/artifacts/`.

Everything new the labs need lives here:
  - pick_device()       : choose cuda / mps / cpu (the repo hardcodes cuda:0)
  - suppress_plots()    : silence the repo attack loops' plt.show()/imshow() spam
  - build_base_params() : load classes.json with DEVICE overridden
  - fgsm() / pgd()       : classic gradient evasion attacks (Lab 1 warm-up)
  - stamp_trigger()      : BadNets visual trigger (Lab 2)
  - poison_dataset()     : trigger + relabel a fraction of training data (Lab 2)
  - fast_train()         : lightweight trainer reusing repo utils (Lab 2)
  - clean_accuracy() / attack_success_rate() : Lab 2 metrics
"""

import os
import sys
import math
import json
import contextlib

# Keep the cloned repo pristine: importing its `utils`/`models` would otherwise
# write a __pycache__/ into the repo dir. Disable bytecode writing before any
# such import happens (set here, which runs before the notebook imports utils).
sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

# --- Locate the cloned repo (sibling folder, one directory up) and make it
# importable. Computed from THIS file's location, so it is independent of the
# notebook's working directory. We never write into REPO_DIR.
LAB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(LAB_DIR, "..", "attacks-on-traffic-sign-recognition"))
ARTIFACT_DIR = os.path.join(LAB_DIR, "artifacts")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Convenient absolute paths to repo inputs (read-only)
CLASSES_JSON = os.path.join(REPO_DIR, "classes.json")
MODELS_DIR = os.path.join(REPO_DIR, "models")
ATTACK_UTILS_DIR = os.path.join(REPO_DIR, "attack_utils")

# Dataset lives in the lab folder by default (keeps the repo untouched), but we
# also detect it if it was placed in the repo's own dataset/ dir.
DATA_DIR = os.path.join(LAB_DIR, "data")
GTSRB_DIR = os.path.join(DATA_DIR, "GTSRB")
_GTSRB_CANDIDATES = [GTSRB_DIR, os.path.join(REPO_DIR, "dataset", "GTSRB")]


# Lab 2 (backdoor) shared constants
BACKDOOR_PATH = os.path.join(ARTIFACT_DIR, "CNNsmallGTSRB_backdoor.pth")
TARGET_CLASS = 14            # Stop
TRIGGER = dict(size=5, color=(255, 0, 255), corner="br", margin=1)  # magenta corner square


def load_gtsrb():
    """Load the GTSRB train/test pickles from the repo's dataset/ dir (read-only).

    Returns (train_data, train_labels, test_data, test_labels) where *_data are
    lists of HxWx3 uint8 arrays and *_labels are lists of ints. Raises a helpful
    error if the dataset has not been downloaded yet."""
    import pickle
    train_pkl = test_pkl = None
    for cand in _GTSRB_CANDIDATES:
        if os.path.exists(os.path.join(cand, "train.pkl")) and \
           os.path.exists(os.path.join(cand, "test.pkl")):
            train_pkl = os.path.join(cand, "train.pkl")
            test_pkl = os.path.join(cand, "test.pkl")
            break
    if train_pkl is None:
        raise FileNotFoundError(
            "GTSRB dataset not found. Download train.pkl + test.pkl into:\n"
            f"    {GTSRB_DIR}\n"
            "(see README_LABS.md for the download link / instructions).")
    with open(train_pkl, "rb") as f:
        tr = pickle.load(f)
    with open(test_pkl, "rb") as f:
        te = pickle.load(f)
    return tr["data"], tr["labels"], te["data"], te["labels"]


# ---------------------------------------------------------------------------
# Device handling — the repo's classes.json hardcodes "cuda:0", which fails on
# a Mac. We never edit classes.json; callers override params['DEVICE'] with this.
# ---------------------------------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Plot suppression — utils.create_noise_mask / train_attack call plt.imshow +
# plt.show() every 100 iterations (hundreds of plots over 20k epochs). This
# context manager neutralises them non-destructively (no edit to utils.py).
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def suppress_plots():
    import matplotlib.pyplot as plt
    saved = {name: getattr(plt, name) for name in ("show", "imshow", "colorbar")}
    # colorbar must also be a no-op: with imshow stubbed there is no mappable,
    # so a real plt.colorbar() (called inside utils.test_attack) would raise.
    plt.show = lambda *a, **k: None
    plt.imshow = lambda *a, **k: None
    plt.colorbar = lambda *a, **k: None
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(plt, name, fn)
        plt.close("all")


# ---------------------------------------------------------------------------
# Params helper — load the repo's classes.json and override DEVICE.
# ---------------------------------------------------------------------------
def build_base_params(dataset="GTSRB"):
    with open(CLASSES_JSON, "r") as f:
        cfg = json.load(f)
    params = dict(cfg)
    params["CLASS_N"] = cfg[dataset]["class_n"]
    params["LABELS"] = cfg[dataset]["labels"]
    params["DEVICE"] = pick_device()
    return params


# ---------------------------------------------------------------------------
# Lab 1 physical-patch attack: assemble the repo's `params` dict (paths +
# hyperparameters) for utils.train_attack / utils.test_attack. Shared by
# precompute_lab1.py and the notebook's optional "run it live" cell.
# 30 km/h sign (source) -> Stop (target class 14), attacking CNNsmall.
#
# Hyperparameters are tuned to RELIABLY flip 30km/h -> Stop on CNN-small: a
# *colored* patch (ATTACK_BETA_1=0, no grayscale penalty) over a larger mask
# (INIT_MASK_THRESHOLD=0.01), with differentiable augmentation on for physical
# robustness. The paper's "inconspicuous" grayscale/small-patch config cannot
# reach a class as visually distant as Stop, so we use this visible variant.
# ---------------------------------------------------------------------------
def lab1_patch_params(out_dir, device=None, attack_epochs=6000, init_mask_epochs=300):
    params = build_base_params("GTSRB")
    params["DEVICE"] = device or pick_device()
    params["MODEL_TYPE"] = "CNNsmall"
    params["PATH_MODEL"] = os.path.join(MODELS_DIR, "CNNsmallGTSRB.pth")
    params["TARGET_CLASS"] = 14  # Stop
    params["PATH_SIGN"] = os.path.join(ATTACK_UTILS_DIR, "30kmh.jpg")
    params["PATH_SIGN_MASK"] = os.path.join(ATTACK_UTILS_DIR, "30kmh_mask.png")
    params["OUTPUT_DIR"] = out_dir
    for key, name in [
        ("PATH_PERT_SIGN_SMALL", "pert_sign_small.png"),
        ("PATH_PERT_SIGN_LARGE", "pert_sign_large.png"),
        ("PATH_ORIG_SIGN_SMALL", "orig_sign_small.png"),
        ("PATH_AUGMENTED_SIGN_SMALL", "augmented_sign_small.png"),
        ("PATH_NOISE_SMALL", "noise_small.png"),
        ("PATH_NOISE_LARGE", "noise_large.png"),
        ("PATH_NOISE_MASK_SMALL", "noise_mask_small.png"),
        ("PATH_NOISE_MASK_LARGE", "noise_mask_large.png"),
        ("PATH_NOISE_TENSOR", "noise_tensor.pt"),
    ]:
        params[key] = os.path.join(out_dir, name)
    # Patch-attack hyperparameters (structure from the repo's 03_Attack_GTSRB.ipynb,
    # values tuned for a reliable 30km/h -> Stop flip on CNN-small).
    params.update({
        "INIT_MASK_THRESHOLD": 0.01, "INIT_MASK_BETA_1": 0.1, "INIT_MASK_BETA_2": 0.01,
        "INIT_MASK_BETA_3": 4, "INIT_MASK_LEARNING_RATE": 0.01,
        "INIT_MASK_EPOCHS": init_mask_epochs, "INIT_MASK_AUGMENTATION": False,
        "ATTACK_LEARNING_RATE": 0.02, "ATTACK_BATCH_SIZE": 8, "ATTACK_BETA_1": 0.0,
        "ATTACK_BETA_2": 0.0, "ATTACK_BETA_3": 4, "ATTACK_EPOCHS": attack_epochs,
        "ATTACK_AUGMENTATION": True,
    })
    return params


# ---------------------------------------------------------------------------
# Lab 1 warm-up: classic gradient evasion attacks on a single image tensor.
# `x` is a CHW float tensor in [0, 1]. Returns an adversarial CHW tensor in
# [0, 1]. CNNsmall returns raw logits, so F.cross_entropy is correct.
# ---------------------------------------------------------------------------
def fgsm(model, x, true_label, eps=0.05, device="cpu", targeted=False, target_class=None):
    """Single-step Fast Gradient Sign Method (L-inf, eps-bounded)."""
    model.eval()
    x_adv = x.clone().detach().unsqueeze(0).to(device).requires_grad_(True)
    if targeted:
        assert target_class is not None
        loss = F.cross_entropy(model(x_adv), torch.tensor([target_class], device=device))
        sign = -1.0  # descend toward the target class
    else:
        loss = F.cross_entropy(model(x_adv), torch.tensor([true_label], device=device))
        sign = +1.0  # ascend away from the true class
    model.zero_grad(set_to_none=True)
    loss.backward()
    x_adv = torch.clamp(x_adv + sign * eps * x_adv.grad.sign(), 0, 1)
    return x_adv.detach().squeeze(0).cpu()  # CPU for display / diffing against the original


def pgd(model, x, label, eps=0.06, alpha=0.01, steps=40, device="cpu",
        targeted=False, target_class=None):
    """Iterative PGD (L-inf). More reliable than FGSM for a *targeted* flip."""
    model.eval()
    x0 = x.clone().detach().unsqueeze(0).to(device)
    x_adv = x0.clone()
    tgt = torch.tensor([target_class if targeted else label], device=device)
    sign = -1.0 if targeted else +1.0
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), tgt)
        model.zero_grad(set_to_none=True)
        loss.backward()
        x_adv = x_adv + sign * alpha * x_adv.grad.sign()
        x_adv = torch.clamp(torch.min(torch.max(x_adv, x0 - eps), x0 + eps), 0, 1)
    return x_adv.detach().squeeze(0).cpu()  # CPU for display / diffing against the original


# ===========================================================================
# Lab 1 (v2): HIGH-RESOLUTION, physically-realistic sticker patch attack.
#
# The victim CNNsmall is hard-wired to a 32x32 input, which is *why* the older
# patch looks pixelated. Real camera->classifier pipelines also downsample a
# detected sign crop to a small classifier input, so instead of optimising at
# 32x32 we optimise a few opaque "stickers" on a crisp HIGH-RES canvas and
# differentiably resize that canvas down to 32x32 for the model. The image the
# audience sees stays sharp; the unmodified model is still fooled.
#
# Faithful to Eykholt et al. (RP2, CVPR 2018): a spatial sticker MASK constrains
# where the patch may go, and Expectation-over-Transformation (random rotation,
# scale, brightness, contrast) makes the patch robust to viewing conditions.
#
# All ops are differentiable. F.interpolate(bilinear) runs natively on MPS;
# F.grid_sample's BACKWARD is not implemented on MPS, so callers must set
# PYTORCH_ENABLE_MPS_FALLBACK=1 before importing torch (that one kernel then
# runs on CPU; the forward and everything else stay on MPS).
# ===========================================================================
def load_highres_sign(path, work_size=256):
    """Load an image and resize to a square work_size canvas. CHW float [0,1]."""
    from PIL import Image
    from torchvision import transforms
    im = Image.open(path).convert("RGB")
    t = transforms.Compose([transforms.Resize((work_size, work_size)),
                            transforms.ToTensor()])
    return torch.clamp(t(im), 0.0, 1.0)


def build_sticker_mask(work_size, rects, channels=1):
    """Binary sticker mask [channels, work_size, work_size] from rectangles.

    `rects` is a list of (x0, y0, x1, y1) given as FRACTIONS in [0, 1] of the
    canvas (so the mask is independent of work_size). Value 1 inside a sticker,
    0 elsewhere -- the RP2-style spatial constraint on where the patch lives.
    """
    m = torch.zeros(1, work_size, work_size)
    for (fx0, fy0, fx1, fy1) in rects:
        x0, x1 = int(round(fx0 * work_size)), int(round(fx1 * work_size))
        y0, y1 = int(round(fy0 * work_size)), int(round(fy1 * work_size))
        m[:, y0:y1, x0:x1] = 1.0
    if channels == 3:
        m = m.expand(3, -1, -1).contiguous()
    return m


def composite_stickers(base, sticker, mask):
    """Opaque overlay: stickers REPLACE the pixels under the mask.

    base/sticker: [3,H,W] in [0,1]; mask: [1or3,H,W] in {0,1}. Differentiable;
    the gradient reaches `sticker` only where mask==1.
    """
    return base * (1.0 - mask) + sticker.clamp(0.0, 1.0) * mask


def _random_affine_theta(n, rot_deg, scale_range, gen, device):
    """n random rotation+uniform-scale affine matrices [n,2,3] for affine_grid."""
    ang = (torch.rand(n, generator=gen) * 2 - 1) * (rot_deg * math.pi / 180.0)
    sc = scale_range[0] + torch.rand(n, generator=gen) * (scale_range[1] - scale_range[0])
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc   # divide by sc -> zoom by sc
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    return theta.to(device)


def _eot_views(img_batch, rot_deg, scale_range, bright, contrast, gen, device):
    """Apply seeded EoT (rotation+scale, brightness, contrast) to a [N,3,H,W] batch.

    Differentiable w.r.t. img_batch (gradients flow back through grid_sample and
    the brightness/contrast math), so this composes inside the attack loss.
    """
    img_batch = img_batch.contiguous()
    n = img_batch.shape[0]
    theta = _random_affine_theta(n, rot_deg, scale_range, gen, device)
    grid = F.affine_grid(theta, list(img_batch.shape), align_corners=False)
    aug = F.grid_sample(img_batch, grid, mode="bilinear",
                        padding_mode="border", align_corners=False)
    b = (torch.rand(n, 1, 1, 1, generator=gen).to(device) - 0.5) * 2 * bright
    c = 1.0 + (torch.rand(n, 1, 1, 1, generator=gen).to(device) - 0.5) * 2 * contrast
    mean = aug.mean(dim=(2, 3), keepdim=True)
    return ((aug - mean) * c + mean + b).clamp(0.0, 1.0)


def _tv(x):
    """Total variation (smoothness / printability) of a CHW tensor."""
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + \
           (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def _gray_penalty(sticker, mask):
    """Penalise colour inside the stickers (push R=G=B for a black/white look)."""
    mu = sticker.mean(0, keepdim=True)
    denom = mask.sum().clamp(min=1.0)
    return (((sticker - mu) ** 2) * mask).sum() / denom


def _blockify(sticker, tile):
    """Quantise a sticker into tile x tile solid blocks (the "printed sticker
    tiles" look) via average-pool + nearest-upsample. Differentiable. tile<=1
    is a no-op (continuous patch)."""
    if not tile or tile <= 1:
        return sticker
    h, w = sticker.shape[-2:]
    s = F.avg_pool2d(sticker.unsqueeze(0), tile, ceil_mode=True)
    return F.interpolate(s, size=(h, w), mode="nearest")[0]  # back to exact size


def _to_grayscale(sticker):
    """Tie the three channels to their mean -> a true black/white sticker."""
    return sticker.mean(0, keepdim=True).expand_as(sticker)


def optimize_sticker_patch(model, base, mask, target_class, device,
                           work_size=256, model_size=32, epochs=300, lr=0.03,
                           eot_samples=8, rot_deg=12.0, scale_range=(0.85, 1.15),
                           bright=0.15, contrast=0.15, reg_tv=1e-3, reg_gray=0.0,
                           tile=1, grayscale=False, seed=0, log_every=50):
    """Optimise opaque stickers (constrained by `mask`) on the high-res `base`
    so the model reads the sign as `target_class`.

    Targeted Expectation-over-Transformation attack: each step composites the
    stickers, builds `eot_samples` randomly augmented high-res views, resizes
    them to model_size with a differentiable bilinear downsample, and minimises
    the cross-entropy toward `target_class` averaged over the views. The model
    only ever sees model_size x model_size, but Adam updates the high-res
    sticker -- so it stays crisp on screen yet fools the small-input model.

    Returns (sticker[3,H,W] cpu, adv_highres[3,H,W] cpu, (idx, conf, probs)),
    where the prediction is the UN-augmented adv canvas evaluated through the
    same bilinear resize used during optimisation.
    """
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)  # seed EoT sampling on CPU
    base = base.to(device)
    mask = mask.to(device)

    sticker = base.clone().detach().to(device).requires_grad_(True)  # neutral start
    if grayscale:
        with torch.no_grad():
            sticker.copy_(_to_grayscale(sticker))
    opt = torch.optim.Adam([sticker], lr=lr)
    tgt = torch.full((eot_samples,), int(target_class), device=device, dtype=torch.long)

    for step in range(epochs):
        opt.zero_grad(set_to_none=True)
        comp = composite_stickers(base, _blockify(sticker, tile), mask)  # [3,H,W]
        batch = comp.unsqueeze(0).expand(eot_samples, -1, -1, -1)   # [N,3,H,W]
        aug = _eot_views(batch, rot_deg, scale_range, bright, contrast, gen, device)
        x_small = F.interpolate(aug, size=(model_size, model_size),
                                mode="bilinear", align_corners=False)
        logits = model(x_small)                                    # raw logits
        loss = F.cross_entropy(logits, tgt)                        # descend toward target
        if reg_tv:
            loss = loss + reg_tv * _tv(sticker * mask)
        if reg_gray:
            loss = loss + reg_gray * _gray_penalty(sticker, mask)
        loss.backward()
        opt.step()
        with torch.no_grad():
            sticker.clamp_(0.0, 1.0)
            if grayscale:
                sticker.copy_(_to_grayscale(sticker))   # keep it black/white
        if log_every and (step % log_every == 0 or step == epochs - 1):
            with torch.no_grad():
                tconf = torch.softmax(logits[0], 0)[int(target_class)].item()
            print(f"  step {step:4d}/{epochs}  loss {loss.item():.4f}  "
                  f"target conf {tconf * 100:5.1f}%")

    with torch.no_grad():
        eff = _blockify(sticker, tile)
        if grayscale:
            eff = _to_grayscale(eff)
        eff_sticker = eff.clamp(0.0, 1.0).detach()
        adv = composite_stickers(base, eff_sticker, mask).detach()
        x_small = F.interpolate(adv.unsqueeze(0), size=(model_size, model_size),
                                mode="bilinear", align_corners=False)[0]
    idx, conf, probs = predict_one(model, x_small, device)
    # return the EFFECTIVE (blockified) sticker so composite_stickers(base,
    # sticker, mask) reproduces `adv` exactly when reloaded.
    return eff_sticker.cpu(), adv.cpu(), (idx, conf, probs)


def robustness_under_augmentation(model, adv_highres, target_class, device,
                                  n=200, model_size=32, rot_deg=12.0,
                                  scale_range=(0.85, 1.15), bright=0.15,
                                  contrast=0.15, batch=50, seed=1234):
    """Fraction of random-augmented, resized-to-model_size views of `adv_highres`
    still predicted as `target_class`. Uses the SAME augmentation family and the
    SAME bilinear resize as optimize_sticker_patch, so it is a faithful estimate
    of physical robustness (does the attack survive viewing-condition changes?)."""
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    img = adv_highres.to(device)
    hits, done = 0, 0
    with torch.no_grad():
        while done < n:
            k = min(batch, n - done)
            b = img.unsqueeze(0).expand(k, -1, -1, -1)
            aug = _eot_views(b, rot_deg, scale_range, bright, contrast, gen, device)
            x_small = F.interpolate(aug, size=(model_size, model_size),
                                    mode="bilinear", align_corners=False)
            hits += int((model(x_small).argmax(1) == int(target_class)).sum().item())
            done += k
    return hits / n


# ---------------------------------------------------------------------------
# Lab 2: BadNets backdoor — trigger stamping, dataset poisoning, fast training,
# and the two evaluation metrics. Data are lists of HxWx3 uint8 numpy arrays
# and int labels, matching the repo's dataset convention; TrafficSignDataset's
# ToTensor() does the single /255 normalisation.
# ---------------------------------------------------------------------------
def stamp_trigger(img, size=5, color=(255, 0, 255), corner="br", margin=1):
    """Return a copy of a HxWx3 uint8 image with a solid square trigger added."""
    out = np.array(img, copy=True)
    h, w = out.shape[:2]
    if corner == "br":
        y0, x0 = h - margin - size, w - margin - size
    elif corner == "bl":
        y0, x0 = h - margin - size, margin
    elif corner == "tr":
        y0, x0 = margin, w - margin - size
    elif corner == "tl":
        y0, x0 = margin, margin
    else:
        raise ValueError(f"unknown corner {corner!r}")
    out[y0:y0 + size, x0:x0 + size] = np.array(color, dtype=out.dtype)
    return out


def poison_dataset(data, labels, p, target=14, trigger_kwargs=None, seed=0):
    """Stamp the trigger on a fraction `p` of samples and relabel them to `target`.

    Returns (new_data, new_labels, poisoned_indices). Inputs are left untouched.
    """
    tk = trigger_kwargs or {}
    rng = np.random.default_rng(seed)
    n = len(data)
    n_pois = int(round(p * n))
    poisoned_idx = set(rng.choice(n, size=n_pois, replace=False).tolist())

    new_data, new_labels = [], []
    for i in range(n):
        if i in poisoned_idx:
            new_data.append(stamp_trigger(data[i], **tk))
            new_labels.append(int(target))
        else:
            new_data.append(np.array(data[i], copy=True))
            new_labels.append(int(labels[i]))
    return new_data, new_labels, sorted(poisoned_idx)


def fast_train(model, train_data, train_labels, test_data, test_labels, device,
               epochs=15, batch=128, lr=1e-3, verbose=True):
    """Lightweight trainer reusing the repo's TrafficSignDataset / model_epoch /
    loss_fun. No x10 augmentation and few epochs -> fast enough for a Mac demo."""
    import utils
    from torch.utils.data.dataloader import DataLoader

    train_set = utils.TrafficSignDataset(train_data, train_labels, device)
    test_set = utils.TrafficSignDataset(test_data, test_labels, device)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(epochs):
        model.train()
        utils.model_epoch(model, train_loader, train=True, optimizer=optimizer, device=device)
        model.eval()
        with torch.no_grad():
            acc, _, _ = utils.model_epoch(model, test_loader, device=device)
        if verbose:
            print(f"  epoch {ep + 1:2d}/{epochs}  clean test acc {float(acc / len(test_set)):.4f}")
    return model


def predict_one(model, x, device):
    """Predict a single CHW float tensor. Returns (class_idx, confidence, prob_vector).

    CNNsmall returns raw logits, so softmax here is correct. Do NOT use this with
    the repo's Transformer (it already returns log_softmax)."""
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x.unsqueeze(0).to(device))[0], 0)
    idx = int(torch.argmax(probs))
    return idx, float(probs[idx]), probs.detach().cpu().numpy()


def _predict_all(model, data, device, batch=256):
    import utils
    from torch.utils.data.dataloader import DataLoader
    ds = utils.TrafficSignDataset(data, [0] * len(data), device)
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(torch.argmax(model(xb.to(device)), dim=1).cpu())
    return torch.cat(preds).numpy()


def clean_accuracy(model, data, labels, device, batch=256):
    """Fraction of CLEAN test images classified correctly (stealth metric)."""
    preds = _predict_all(model, data, device, batch)
    return float((preds == np.asarray(labels)).mean())


def attack_success_rate(model, data, labels, device, target=14, trigger_kwargs=None, batch=256):
    """Fraction of TRIGGERED, non-target test images predicted as `target`."""
    tk = trigger_kwargs or {}
    labels = np.asarray(labels)
    keep = np.flatnonzero(labels != target)
    trig = [stamp_trigger(data[i], **tk) for i in keep]
    preds = _predict_all(model, trig, device, batch)
    return float((preds == target).mean())
