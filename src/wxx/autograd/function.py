
import weakref

from .. import Tensor
from ..backend import context

class Function:
    """Function that supports autograd"""

    def __init__(self, *inputs):
        if context.MODE == 0:
            self.inputs = [input if not None else None for input in inputs]
            self.needs_input_grad = [
                input.requires_grad if isinstance(input, Tensor) and not None else False
                for input in inputs
            ]
        else:
            self.inputs = [input if not None else None for input in inputs]
            self.needs_input_grad = [
                False if isinstance(input, Tensor) and not None else False
                for input in inputs
            ]

        NotImplemented

    def forward(self, *args):
        NotImplemented

    def backward(self, grad_output):
        NotImplemented

    def hold_output(self, out):
        """Remember ``out`` without creating a reference cycle.

        The output tensor already points back here through
        ``Tensor.creator``, so storing a second strong reference makes the
        pair ``{out, function}`` a cycle.  Cyclic garbage is only reclaimed
        by a generation-2 collection, which Python 3.14 no longer schedules
        automatically, so unlinking the graph would strand GPU buffers for
        the whole run.  A weak reference keeps the tensor reachable from the
        function during ``backward`` without owning it.
        """
        self._out_ref = weakref.ref(out)

    def held_output(self):
        """Return the tensor passed to :meth:`hold_output` and drop the ref."""
        ref = getattr(self, "_out_ref", None)
        self._out_ref = None
        return None if ref is None else ref()
