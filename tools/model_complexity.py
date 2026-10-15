"""Optional generator parameter, FLOP, and inference-time reporting."""

from time import perf_counter

import numpy as np
import torch
from thop import profile


def evaluate_model_complexity(model, device, resolution=(256, 256)):
    """Print model complexity without being part of the training path."""
    was_training = model.training
    model.eval()
    optical = torch.randn(1, 3, *resolution, device=device)
    edge = torch.randn(1, 2, *resolution, device=device)
    sar = torch.randn(1, 2, *resolution, device=device)
    block = torch.randn(1, 2, *resolution, device=device)
    inputs = (optical, block, torch.cat((sar, edge), dim=1))

    flops, parameters = profile(model, inputs=inputs, verbose=False)
    print(f"Generator FLOPs: {flops / 1e9:.2f} G")
    print(f"Generator parameters: {parameters / 1e6:.2f} M")

    timings = []
    with torch.no_grad():
        for _ in range(50):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(100):
            start = perf_counter()
            model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((perf_counter() - start) * 1000)

    average_ms = float(np.mean(timings))
    print(f"Average inference time: {average_ms:.2f} ms")
    print(f"Estimated throughput: {1000.0 / average_ms:.2f} images/s")
    model.train(was_training)

