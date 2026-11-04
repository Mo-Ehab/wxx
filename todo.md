# wxx TODO

## Done (2026 cleanup)
- Consistent validation/shape checks across tensor ops.
- Lint clean (ruff, line-length 100, E/F/W).
- Hyena paper-only architecture.
- GPT-2 and ResNet example models.
- Shader reference lint test.
- Removed unused shaders (`fft`, `ifft`, legacy `sinc`/`sincdx`).
- Merged the `equal*` shader family into one mode-driven `helpers/equal.wgsl`.
- Merged `max`/`min` into one op-selector `helpers/reduce_minmax.wgsl`.
- Added autograd-aware `var`/`std` (`Tensor.var`/`Tensor.std`).
- Added `AdaptiveAvgPool2d` + fused forward/backward kernels; ResNet uses it.
- Added learning-rate schedulers (`LambdaLR`, `StepLR`, `ExponentialLR`,
  `CosineAnnealingLR`, `WarmupCosineLR`).
- Added GPT-2/ResNet state-dict round-trip tests.
- GPT-2 applies a cached causal mask; added `GPT2.generate()` for sampling.
- Added `Tensor.detach()` / `detach_()` (view that cuts the autograd graph).
- Added `Conv2DTranspose` (forward / dx / dw kernels, autograd, module).
- Fixed `Conv2D` bias gradients (`sum(axis=(0, 2, 3))` is unsupported; reduce in two steps).

## Planned
- Add model zoo / example training scripts for GPT-2 and ResNet.
- Add mixed-precision (fp16) support.
- Add gradient checkpointing for large models.
- Add fused variance/std kernels (currently a short chain of pointwise ops).
