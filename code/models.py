"""Model zoo (Paper A) -- all off-the-shelf torchvision models, no new modules designed.

Design principles (matching the hard constraints in PROJECT.md):
  * use only official torchvision implementations and official pretrained weights, no custom networks
  * cover 4 families so that the "efficiency-accuracy frontier" is sampled evenly
  * when weights cannot be downloaded, fall back to random initialization (weights=None) rather than aborting

Usage:
    from models import build_model, MODEL_ZOO
    m = build_model('resnet18', num_classes=6, pretrained=True)
"""

import torch
import torch.nn as nn
import torchvision

# Family split -- the grouping basis for the "efficiency-accuracy frontier" in the paper
MODEL_ZOO = {
    # ---- classic CNNs ----
    "resnet18":           ("cnn_classic",  torchvision.models.resnet18,        "IMAGENET1K_V1"),
    "resnet50":           ("cnn_classic",  torchvision.models.resnet50,        "IMAGENET1K_V2"),
    "densenet121":        ("cnn_classic",  torchvision.models.densenet121,     "IMAGENET1K_V1"),
    # ---- lightweight / mobile ----
    "mobilenet_v3_small": ("lightweight",  torchvision.models.mobilenet_v3_small, "IMAGENET1K_V1"),
    "mobilenet_v3_large": ("lightweight",  torchvision.models.mobilenet_v3_large, "IMAGENET1K_V2"),
    "shufflenet_v2_x1_0": ("lightweight",  torchvision.models.shufflenet_v2_x1_0, "IMAGENET1K_V1"),
    "efficientnet_b0":    ("lightweight",  torchvision.models.efficientnet_b0,  "IMAGENET1K_V1"),
    "regnet_x_400mf":     ("lightweight",  torchvision.models.regnet_x_400mf,   "IMAGENET1K_V1"),
    # ---- modern CNNs ----
    "convnext_tiny":      ("modern_cnn",   torchvision.models.convnext_tiny,    "IMAGENET1K_V1"),
    # ---- Transformers ----
    "vit_b_16":           ("transformer",  torchvision.models.vit_b_16,         "IMAGENET1K_V1"),
    "swin_t":             ("transformer",  torchvision.models.swin_t,           "IMAGENET1K_V1"),
}

DEFAULT_MODELS = list(MODEL_ZOO.keys())


def _replace_head(model, name, num_classes):
    """Replace the classification head with a num_classes output. The head attribute name differs per model."""
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name.startswith("mobilenet_v3"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name.startswith("shufflenet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("efficientnet"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name.startswith("regnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("convnext"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name.startswith("vit"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif name.startswith("swin"):
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError("no classification-head replacement rule registered for model: %s" % name)
    return model


def build_model(name, num_classes=6, pretrained=True, verbose=True):
    """Build a model.

    With pretrained=True, try to load the ImageNet weights; if there is no local cache and the
    external network is unreachable, fall back to random initialization and flag it in the log.
    """
    if name not in MODEL_ZOO:
        raise KeyError("unknown model %r, choices: %s" % (name, ", ".join(MODEL_ZOO)))

    family, ctor, weight_tag = MODEL_ZOO[name]
    weights = None
    got_pretrained = False

    if pretrained:
        try:
            weights = torchvision.models.get_model_weights(name)[weight_tag]
        except Exception as e:
            if verbose:
                print("[warn] failed to get the weights enum %s: %s" % (name, e), flush=True)
            weights = None

    try:
        model = ctor(weights=weights)
        got_pretrained = weights is not None
    except Exception as e:
        # Typical case: no external network, torchvision download fails
        if verbose:
            print("[warn] %s failed to load pretrained weights (most likely no external network), falling back to random init: %s"
                  % (name, str(e)[:200]), flush=True)
        model = ctor(weights=None)

    model = _replace_head(model, name, num_classes)

    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print("[model] %s family=%s params=%.2fM pretrained=%s"
              % (name, family, n / 1e6, got_pretrained), flush=True)

    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for m in DEFAULT_MODELS:
        try:
            net = build_model(m, num_classes=6, pretrained=False, verbose=False)
            n = count_params(net)
            print("%-22s %-14s %8.2f M" % (m, MODEL_ZOO[m][0], n / 1e6))
        except Exception as e:
            print("%-22s FAILED: %s" % (m, e))
