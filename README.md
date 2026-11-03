# wxx — WebGPU tensor library

`wxx` is a small PyTorch-like tensor library that runs on the GPU through
WebGPU (via `wgpu-py`).  It provides tensors, autograd, common neural-network
modules, FFT primitives and a few example models.

## Install

```bash
pip install git+https://github.com/Mo-Ehab/wxx
```

## Quick start

```python
from wxx import Tensor, functional

a = Tensor((1,6), value = 10)
b = a + 1.0
print(b)
print(b.shape, b.is_contiguous())
```

## Models

```python
from wxx.nn.gpt import GPT2
from wxx.nn.resnet import ResNet18
from wxx.nn.hyena import Hyena

gpt = GPT2(vocab_size=1024, d_model=256, n_heads=8, n_layers=6)
resnet = ResNet18(num_classes=10)
hyena = Hyena(vocab_size=1024, d_model=256, n_layers=6, order=4)
```

## Training sketch

```python
from wxx.nn.loss import CCELoss
from wxx.nn.optimizer import AdamW
from wxx.nn.scheduler import WarmupCosineLR
from wxx.nn import clip_grad_norm_

optim = AdamW(model.parameters(), lr=1e-3)
scheduler = WarmupCosineLR(optim, warmup_steps=100, total_steps=5000, min_lr=1e-5)
loss_fn = CCELoss()

out = model(x)
b, s, v = out.shape
loss = loss_fn(out.reshape((b * s, v)), y.reshape((b * s,)))
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optim.step()
scheduler.step()
optim.zero_grad()
```

Other schedules: `LambdaLR`, `StepLR`, `ExponentialLR`, `CosineAnnealingLR`.

## Save / load

```python
from wxx import save, load
save(model, "model.npz")
load("model.npz", model)
```

## Tests

```bash
pip install -e "wxx[test] @ git+https://github.com/Mo-Ehab/wxx" --src .

pytest
```


## Layout

- `wxx/Tensor.py` — core tensor, views, reshape/permute/...
- `wxx/autograd/` — autograd functions
- `wxx/functional/` — GPU kernels (pointwise, reductions, conv, FFT, ...)
- `wxx/nn/` — modules (Linear, Conv2D, Attention, Hyena, GPT2, ResNet, ...)
- `wxx/nn/module.py` — base `Module`, reusable layers/activations/pooling
- `wxx/nn/scheduler.py` — learning-rate schedulers
- `wxx/validation.py` — shape/dimension/layout validators
- `wxx/shaders/` — WGSL shaders
