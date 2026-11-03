import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wxx.nn.scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LambdaLR,
    StepLR,
    WarmupCosineLR,
)


class DummyOptimizer:
    def __init__(self, lr):
        self.lr = lr


def test_warmup_cosine():
    opt = DummyOptimizer(1e-3)
    sched = WarmupCosineLR(opt, warmup_steps=4, total_steps=10, min_lr=1e-5)

    # The constructor sets step 0 = first warmup value.
    assert abs(opt.lr - 1e-3 / 4) < 1e-12
    for _ in range(3):
        sched.step()
    assert abs(opt.lr - 1e-3) < 1e-12

    while sched.last_epoch < 9:
        sched.step()
    assert abs(opt.lr - 1e-5) < 1e-9
    assert abs(sched.get_last_lr() - 1e-5) < 1e-9
    sched.step()
    assert abs(opt.lr - 1e-5) < 1e-9


def test_warmup_cosine_zero_warmup():
    opt = DummyOptimizer(1e-3)
    sched = WarmupCosineLR(opt, warmup_steps=0, total_steps=5, min_lr=0.0)
    assert abs(opt.lr - 1e-3) < 1e-12
    while sched.last_epoch < 4:
        sched.step()
    assert abs(opt.lr) < 1e-9


def test_cosine_annealing():
    opt = DummyOptimizer(1.0)
    sched = CosineAnnealingLR(opt, T_max=4, eta_min=0.1)
    assert abs(opt.lr - 1.0) < 1e-12
    for _ in range(4):
        sched.step()
    assert abs(opt.lr - 0.1) < 1e-9


def test_step_and_exponential():
    opt = DummyOptimizer(1.0)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    assert abs(opt.lr - 1.0) < 1e-12
    for _ in range(4):
        sched.step()
    assert abs(opt.lr - 0.25) < 1e-12

    opt2 = DummyOptimizer(1.0)
    sched2 = ExponentialLR(opt2, gamma=0.5)
    for _ in range(3):
        sched2.step()
    assert abs(opt2.lr - 0.125) < 1e-12


def test_lambda_and_state_dict():
    opt = DummyOptimizer(2.0)
    sched = LambdaLR(opt, lambda step: 1.0 / (step + 1))
    assert abs(opt.lr - 2.0) < 1e-12
    sched.step()
    assert abs(opt.lr - 1.0) < 1e-12

    state = sched.state_dict()
    restored = LambdaLR(DummyOptimizer(2.0), lambda step: 1.0 / (step + 1))
    restored.load_state_dict(state)
    assert restored.last_epoch == sched.last_epoch
    assert abs(restored.get_last_lr() - sched.get_last_lr()) < 1e-12


TESTS = [
    ("warmup_cosine", test_warmup_cosine),
    ("warmup_cosine_zero_warmup", test_warmup_cosine_zero_warmup),
    ("cosine_annealing", test_cosine_annealing),
    ("step_and_exponential", test_step_and_exponential),
    ("lambda_and_state_dict", test_lambda_and_state_dict),
]

if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
