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
- Added differentiable `Tensor.diagonal(offset, dim1, dim2)` view over any two dims.
- Added `avgpool1d` / `maxpool1d` (fwd + dx kernels) and `nn.AvgPool1d` / `nn.MaxPool1d`.
- Added `return_indices` to `avgpool1d`/`avgpool2d`/`maxpool1d`/`maxpool2d` and their modules
  (`u32` flat input indices; no window maximum maps to `0xFFFFFFFF`).
- Fixed pooling output sizes to PyTorch's `(in + 2*pad - kernel) // stride + 1`
  (`avgpool2d`/`maxpool2d` previously produced one extra row/column when the window did not align).
- Added differentiable `Tensor.unfold(dimension, size, step)` sliding-window view + scatter-add backward.
- Fixed `Tensor.permute((...))` (axes passed as one tuple) failing in backward with
  `TypeError`; autograd now flattens and normalises the axes before storing them.

## Planned
- Add model zoo / example training scripts for GPT-2 and ResNet.
- Add mixed-precision (fp16) support.
- Add gradient checkpointing for large models.
- Add fused variance/std kernels (currently a short chain of pointwise ops).
