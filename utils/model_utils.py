"""Small model helpers shared by the generator and discriminator."""

import torch.nn as nn


def weights_init(init_type="normal", gain=0.02):
    """Return an initializer compatible with ``nn.Module.apply``."""
    def init_func(module):
        class_name = module.__class__.__name__
        if hasattr(module, "weight") and (
            "Conv" in class_name or "Linear" in class_name
        ):
            if init_type == "normal":
                nn.init.normal_(module.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(module.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(module.weight.data, gain=gain)

            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias.data, 0.0)
        elif "BatchNorm2d" in class_name:
            nn.init.normal_(module.weight.data, 1.0, gain)
            nn.init.constant_(module.bias.data, 0.0)

    return init_func


def spectral_norm(module, enabled=True):
    """Apply spectral normalization when requested."""
    return nn.utils.spectral_norm(module) if enabled else module


def extract_patches(x, kernel_size=3, stride=1):
    """Extract sliding spatial patches in the layout expected by CCFS."""
    if kernel_size != 1:
        x = nn.ZeroPad2d(1)(x)
    x = x.permute(0, 2, 3, 1)
    return x.unfold(1, kernel_size, stride).unfold(
        2, kernel_size, stride
    ).contiguous()


def requires_grad(model, flag=True):
    """Enable or disable gradients for every model parameter."""
    for parameter in model.parameters():
        parameter.requires_grad = flag
