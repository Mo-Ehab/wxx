"""Dense tensors backed by WebGPU storage buffers.

A :class:`Tensor` stores a logical ``shape`` (a list, kept for compatibility
with the kernels), a row-major ``stride`` (in elements), an ``offset`` into the
buffer and a ``type`` of ``"f32"``, ``"u32"`` or ``"c64"``.

Views (slicing, ``narrow``, ``permute``, ...) share the parent buffer and carry
their own offset/stride.  Operations that cannot express a layout in a shader
materialise a contiguous copy instead.
"""


import numpy as np
import wgpu

from . import validation
from .backend.device import get_buffer_pool, get_cmd_buffer, get_device

device = get_device()
cmd_buf = get_cmd_buffer()
buffer_pool = get_buffer_pool()

_DTYPES = ("f32", "u32", "c64")


class Tensor:
    """N-dimensional tensor backed by a WebGPU storage buffer."""

    def __init__(
        self,
        shape,
        buffer=None,
        data=None,
        value=1.0,
        offset=0,
        stride=None,
        requires_grad=False,
        dtype=None,
    ):
        shape = validation.normalize_shape(shape)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError(f"offset must be a non-negative int, got {offset!r}")
        if dtype is not None and dtype not in _DTYPES:
            raise ValueError(f"unsupported dtype {dtype!r}; expected one of {_DTYPES}")

        self.shape = list(shape)
        self.offset = offset
        self._base = None
        self.type = dtype if dtype is not None else "f32"

        if stride is None:
            self.stride = validation.contiguous_strides(self.shape)
        else:
            stride = tuple(stride)
            if len(stride) != len(self.shape):
                raise ValueError(
                    f"stride {stride} must have one entry per shape dimension "
                    f"{tuple(self.shape)}"
                )
            self.stride = stride

        self.nbytes = validation.numel(self.shape) * self.itemsize()

        if data is None and buffer is None:
            if self.type == "c64":
                self.data = np.zeros(
                    validation.numel(self.shape) * 2, dtype=np.float32
                ).tobytes()
            else:
                self.data = np.full(
                    validation.numel(self.shape), value, dtype=np.float32
                ).tobytes()
        else:
            self.data = data

        self.buffer = (
            buffer_pool.create_buffer_with_data(
                data=bytes(self.data),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
                label="Tensor",
            )
            if buffer is None
            else buffer
        )
        self.requires_grad = requires_grad
        self.creator = None
        self.grad = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def itemsize(self):
        """Bytes per logical element (complex64 stores two float32s)."""
        return 8 if self.type == "c64" else 4

    def calc_contiguous_strides(self):
        """Row-major strides for the current shape."""
        return validation.contiguous_strides(self.shape)

    def is_contiguous(self):
        """True for a compact row-major tensor starting at buffer offset 0.

        Views with a non-zero offset are treated as non-contiguous even when
        their strides are compact.  That is conservative: every kernel already
        addresses ``offset`` explicitly, and a copy is always correct.
        """
        if self.offset != 0:
            return False
        return validation.is_contiguous_layout(self.shape, self.stride)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        # Expand a single Ellipsis into the required number of full slices.
        if any(item is Ellipsis for item in index):
            ellipsis_index = next(
                i for i, item in enumerate(index) if item is Ellipsis
            )
            missing = len(self.shape) - (len(index) - 1)
            if missing < 0:
                raise IndexError("too many indices for tensor")
            index = (
                index[:ellipsis_index]
                + (slice(None),) * missing
                + index[ellipsis_index + 1 :]
            )

        # Normalise list/array indices into tensors (advanced indexing).
        index_list = []
        has_tensor_index = False
        for item in index:
            if isinstance(item, (list, np.ndarray)):
                from .functional.utils import from_numpy

                item = (
                    item
                    if hasattr(item, "buffer")
                    else from_numpy(np.asarray(item))
                )
            if hasattr(item, "buffer"):
                has_tensor_index = True
            index_list.append(item)
        index = tuple(index_list)

        if has_tensor_index:
            return self._advanced_getitem(index)

        if len(index) < len(self.shape):
            index = index + (slice(None),) * (len(self.shape) - len(index))
        if len(index) > len(self.shape):
            raise IndexError("too many indices for tensor")

        from .backend import context

        if self.requires_grad and context.MODE == 0:
            from wxx import autograd

            return autograd.getitem(self).forward(self, index)

        return self._basic_getitem(index)

    def _basic_getitem(self, index):
        """Zero-copy view used by both regular and autograd indexing."""
        new_offset = self.offset
        new_shape = list(self.shape)
        new_stride = list(self.stride)

        for axis, item in enumerate(index):
            if isinstance(item, slice):
                start, stop, step = item.indices(self.shape[axis])
                new_offset += start * self.stride[axis]
                length = max(
                    0,
                    (stop - start + (step - 1 if step > 0 else step + 1)) // step,
                )
                new_shape[axis] = length
                new_stride[axis] = self.stride[axis] * step
            elif isinstance(item, int) and not isinstance(item, bool):
                position = item if item >= 0 else self.shape[axis] + item
                if not 0 <= position < self.shape[axis]:
                    raise IndexError(
                        f"index {item} out of bounds for axis {axis} "
                        f"with size {self.shape[axis]}"
                    )
                new_offset += position * self.stride[axis]
                new_shape[axis] = None  # mark for removal
                new_stride[axis] = None  # mark for removal
            else:
                raise IndexError(f"unsupported index type {type(item).__name__}")

        final_shape = [size for size in new_shape if size is not None]
        final_stride = [step for step in new_stride if step is not None]

        out = Tensor(
            shape=final_shape,
            buffer=self.buffer,
            stride=final_stride,
            offset=new_offset,
            requires_grad=self.requires_grad,
            dtype=self.type,
        )
        out._base = self
        return out

    def _advanced_getitem(self, index):
        from .functional.utils import from_numpy

        if not self.is_contiguous():
            tensor_self = self.contiguous()
        else:
            tensor_self = self

        first = index[0]
        if isinstance(first, (bool, np.bool_)):
            mask = np.asarray(first)
            indices = np.flatnonzero(mask).astype(np.int32)
            result = tensor_self.numpy().reshape(-1)[indices]
            out = from_numpy(result)
            out.type = self.type
            out.nbytes = validation.numel(out.shape) * out.itemsize()
            out.requires_grad = self.requires_grad
            return out

        np_indices = []
        for item in index:
            if hasattr(item, "numpy"):
                np_indices.append(item.numpy().astype(np.int32))
            else:
                np_indices.append(item)

        gathered = tensor_self.numpy()[tuple(np_indices)]
        out = from_numpy(gathered)
        out.type = self.type
        out.nbytes = validation.numel(out.shape) * out.itemsize()
        out.requires_grad = self.requires_grad
        return out

    def __setitem__(self, index, value):
        from . import functional

        functional.assign(self[index], value)

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    def narrow(self, dim, start, length):
        """Return a view of ``length`` elements along ``dim`` starting at ``start``."""
        rank = len(self.shape)
        dim = validation.normalize_dim(dim, rank)
        if not isinstance(start, int) or isinstance(start, bool):
            raise TypeError(f"start must be an int, got {type(start).__name__}")
        if not isinstance(length, int) or isinstance(length, bool):
            raise TypeError(f"length must be an int, got {type(length).__name__}")
        if start < 0 or length < 0 or start + length > self.shape[dim]:
            raise IndexError(
                f"narrow bounds out of range: start={start}, length={length}, "
                f"dim {dim} has size {self.shape[dim]}"
            )

        new_offset = self.offset + start * self.stride[dim]
        new_shape = list(self.shape)
        new_shape[dim] = length

        out = Tensor(
            shape=new_shape,
            buffer=self.buffer,
            stride=self.stride,
            offset=new_offset,
            requires_grad=self.requires_grad,
            dtype=self.type,
        )
        out._base = self
        return out

    def slice(self, dim, start, length):
        """Differentiable :meth:`narrow`."""
        from wxx import autograd

        return autograd.narrow(self).forward(self, dim, start, length)

    def chunk(self, chunks, dim=0):
        """Split ``dim`` into ``chunks`` equal parts (differentiable)."""
        rank = len(self.shape)
        dim = validation.normalize_dim(dim, rank, name="dim")
        if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks < 1:
            raise ValueError(f"chunks must be a positive int, got {chunks!r}")
        size = self.shape[dim]
        if size % chunks != 0:
            raise ValueError(
                f"cannot split dimension {dim} of size {size} into "
                f"{chunks} equal chunks"
            )
        step = size // chunks
        from wxx import autograd

        return tuple(
            autograd.narrow(self).forward(self, dim, i * step, step)
            for i in range(chunks)
        )

    def reshape(self, *shape):
        """Return a tensor with the same elements and a new ``shape``."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        from wxx import autograd

        return autograd.reshape(self).forward(self, shape)

    def view(self, *shape):
        """Alias of :meth:`reshape` (may copy a non-contiguous tensor)."""
        return self.reshape(*shape)

    def permute(self, *axis):
        """Return a view with its dimensions reordered."""
        from wxx import autograd

        return autograd.permute(self).forward(self, axis)

    def transpose(self, dim0, dim1):
        """Swap two dimensions (a view)."""
        rank = len(self.shape)
        dim0 = validation.normalize_dim(dim0, rank, name="dim0")
        dim1 = validation.normalize_dim(dim1, rank, name="dim1")
        axis = list(range(rank))
        axis[dim0], axis[dim1] = axis[dim1], axis[dim0]
        from wxx import autograd

        return autograd.permute(self).forward(self, axis)

    def t(self):
        """Swap the last two dimensions (a view)."""
        if len(self.shape) < 2:
            raise IndexError("t() expects a tensor with at least 2 dimensions")
        return self.transpose(-1, -2)

    def contiguous(self):
        """Return ``self`` when contiguous, else a materialised copy."""
        from . import functional

        return functional.contiguous(self)

    def unsqueeze(self, dim):
        """Insert a size-1 dimension at ``dim``."""
        from wxx import autograd

        return autograd.unsqueeze(self).forward(self, dim)

    def squeeze(self, dim=None):
        """Remove size-1 dimensions (all of them when ``dim`` is ``None``)."""
        from wxx import autograd

        return autograd.squeeze(self).forward(self, dim)

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten dimensions ``start_dim..end_dim`` into one."""
        from wxx import autograd

        return autograd.flatten(self).forward(self, start_dim, end_dim)

    # ------------------------------------------------------------------
    # Autograd entry point
    # ------------------------------------------------------------------
    def backward(self):
        """Backpropagate from this tensor through its recorded graph."""
        from . import functional
        from .backend.context import inferenceMode

        if not self.requires_grad:
            raise RuntimeError(
                "cannot call backward on a tensor that does not require gradients"
            )
        if self.grad is None:
            self.grad = Tensor(shape=self.shape, dtype=self.type)

        topo = []
        visited = set()

        def build_topo(tensor):
            tensor_id = id(tensor)
            if (
                tensor_id not in visited
                and isinstance(tensor, Tensor)
                and tensor.requires_grad
            ):
                visited.add(tensor_id)
                if tensor.creator is not None:
                    for parent in tensor.creator.inputs:
                        if parent is not None and parent.requires_grad:
                            build_topo(parent)
                topo.append(tensor)

        build_topo(self)

        with inferenceMode():
            gradients = {id(self): self.grad}

            while topo:
                tensor = topo.pop()
                grad_output = gradients.pop(id(tensor), None)
                if grad_output is None:
                    # Nothing flows through this node: it sits behind an op
                    # whose output does not require grad, or the node does not
                    # contribute to the loss.  Unlink it so its activations can
                    # be reclaimed by refcounting instead of waiting for a
                    # generation-2 GC pass.
                    if tensor.creator is not None:
                        tensor.creator.inputs = None
                        tensor.creator = None
                    continue

                fn = tensor.creator
                if fn is not None:
                    parent_grads = fn.backward(grad_output)
                    if len(fn.inputs) == 1:
                        parent_grads = (parent_grads,)
                    elif not isinstance(parent_grads, (tuple, list)):
                        raise TypeError(
                            f"{type(fn).__name__}.backward must return one "
                            "gradient per input"
                        )
                    if len(parent_grads) != len(fn.inputs):
                        raise RuntimeError(
                            f"{type(fn).__name__}.backward returned "
                            f"{len(parent_grads)} gradients for "
                            f"{len(fn.inputs)} inputs"
                        )

                    for parent, grad in zip(fn.inputs, parent_grads):
                        if parent is None or not parent.requires_grad or grad is None:
                            continue

                        accumulated = gradients.get(id(parent))
                        if accumulated is None:
                            gradients[id(parent)] = grad
                        elif getattr(accumulated, "type", "f32") == "c64":
                            functional.complex_add_inplace(accumulated, grad)
                        else:
                            functional._pointwise(accumulated, grad, 0)

                    fn.inputs = None
                    tensor.creator = None

                if fn is None and tensor is not self:
                    if tensor.grad is not None:
                        if getattr(tensor.grad, "type", "f32") == "c64":
                            functional.complex_add_inplace(tensor.grad, grad_output)
                        else:
                            functional._pointwise(tensor.grad, grad_output, 0)
                    else:
                        tensor.grad = grad_output

    # ------------------------------------------------------------------
    # Host access
    # ------------------------------------------------------------------
    def numpy(self):
        """Copy this tensor to a NumPy array (synchronises the GPU)."""
        self.read_buffer()
        if self.type == "f32":
            dtype = np.float32
        elif self.type == "u32":
            dtype = np.uint32
        elif self.type == "c64":
            dtype = np.complex64
        else:
            raise TypeError(f"unsupported tensor type {self.type!r}")

        nbytes = validation.numel(self.shape) * self.itemsize()
        if self.is_contiguous():
            return (
                np.frombuffer(self.data[:nbytes], dtype=dtype)
                .copy()
                .reshape(tuple(self.shape))
            )

        data = np.frombuffer(self.data, dtype=dtype).copy()
        byte_stride = tuple(step * data.itemsize for step in self.stride)
        view = np.lib.stride_tricks.as_strided(
            data[self.offset :],
            shape=self.shape,
            strides=byte_stride,
            check_bounds=True,
        )
        return view.copy()

    def read_buffer(self):
        """Read the whole backing buffer back into ``self.data``."""
        cmd_buf.flush()
        buffsize = self.buffer.size

        buf_readback = buffer_pool.create_buffer(
            size=buffsize,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

        encoder = device.create_command_encoder()
        encoder.copy_buffer_to_buffer(self.buffer, 0, buf_readback, 0, buffsize)
        device.queue.submit([encoder.finish()])
        buf_readback.map_sync(wgpu.MapMode.READ)
        cmd_buf.complete_pending()
        self.data = buf_readback.read_mapped()
        buf_readback.unmap()
        buffer_pool.release(buf_readback)

    def __repr__(self):
        array = self.numpy()
        array_str = np.array2string(
            array, prefix="tensor(", precision=4, suppress_small=True
        )
        return (
            f"tensor({array_str}, shape={self.shape}), "
            f"require grad={self.requires_grad}"
        )

    def item(self):
        """Return the value of a one-element tensor as a Python scalar."""
        if validation.numel(self.shape) != 1:
            raise RuntimeError(
                "cannot convert a tensor with more than one element to a scalar"
            )
        return self.numpy().item()

    def destroy(self):
        """Release the backing buffer (views defer to their base)."""
        if getattr(self, "buffer", None) is not None and self._base is None:
            buffer = self.buffer
            self.buffer = None
            cmd_buf.defer_destroy(buffer)

    def __del__(self):
        try:
            if getattr(self, "_base", None) is None:
                self.destroy()
        except BaseException:
            pass

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other):
        from wxx import autograd

        if not isinstance(other, Tensor):
            return autograd.addscaler(self).forward(self, float(other))
        return autograd.add(self, other).forward(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from wxx import autograd

        if not isinstance(other, Tensor):
            return autograd.subscaler(self).forward(self, float(other))
        return autograd.sub(self, other).forward(self, other)

    def __neg__(self):
        from wxx import autograd

        return autograd.negate(self).forward(self)

    def __mul__(self, other):
        from wxx import autograd

        if not isinstance(other, Tensor):
            return autograd.mulscaler(self).forward(self, float(other))
        return autograd.mul(self, other).forward(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from wxx import autograd

        if not isinstance(other, Tensor):
            return autograd.mulscaler(self).forward(self, 1.0 / float(other))
        return autograd.div(self, other).forward(self, other)

    def __matmul__(self, other):
        from wxx import autograd

        return autograd.matmul(self, other).forward(self, other)

    def matmul(self, other):
        return self.__matmul__(other)

    def __pow__(self, other):
        from wxx import autograd

        if isinstance(other, (int, float)) and other == 2:
            return autograd.square(self).forward(self)
        if isinstance(other, (int, float)):
            return autograd.powscaler(self).forward(self, float(other))
        raise TypeError(
            f"tensor exponents are not supported, got {type(other).__name__}"
        )

    def __eq__(self, other):
        from . import functional

        return functional.equal(self, other)

    def matmul_add(self, other, bias):
        from wxx import autograd

        return autograd.matmuladd(self, other, bias).forward(self, other, bias)

    # ------------------------------------------------------------------
    # Unary / reduction methods
    # ------------------------------------------------------------------
    def relu(self):
        from wxx import autograd

        return autograd.relu(self).forward(self)

    def gelu(self):
        from wxx import autograd

        return autograd.gelu(self).forward(self)

    def sin(self):
        from wxx import autograd

        return autograd.sin(self).forward(self)

    def sigmoid(self):
        from wxx import autograd

        return autograd.sigmoid(self).forward(self)

    def silu(self):
        from wxx import autograd

        return autograd.sigmoid(self).forward(self) * self

    def selu(self):
        from wxx import autograd

        return autograd.selu(self).forward(self)

    def softmax(self, dim=-1):
        from wxx import autograd

        return autograd.softmax(self).forward(self, dim)

    def square(self):
        from wxx import autograd

        return autograd.square(self).forward(self)

    def exp(self):
        from wxx import autograd

        return autograd.exp(self).forward(self)

    def sqrt(self):
        from wxx import autograd

        return autograd.sqrt(self).forward(self)

    def rsqrt(self):
        from wxx import autograd

        return autograd.rsqrt(self).forward(self)

    def log(self):
        from wxx import autograd

        return autograd.log(self).forward(self)

    def clamp(self, min=np.finfo(np.float32).min, max=np.finfo(np.float32).max):
        from wxx import autograd

        return autograd.clamp(self).forward(self, min, max)

    def sum(self, axis=None, keepdims=False, *, keepdim=None):
        from wxx import autograd

        if keepdim is not None:
            keepdims = keepdim
        return autograd.sum(self).forward(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False, *, keepdim=None):
        from wxx import autograd

        if keepdim is not None:
            keepdims = keepdim
        return autograd.mean(self).forward(self, axis, keepdims)

    def var(self, axis=None, keepdims=False, *, keepdim=None):
        from wxx import autograd

        if keepdim is not None:
            keepdims = keepdim
        return autograd.var(self).forward(self, axis, keepdims)

    def std(self, axis=None, keepdims=False, *, keepdim=None):
        from wxx import autograd

        if keepdim is not None:
            keepdims = keepdim
        return autograd.std(self).forward(self, axis, keepdims)

    def abs(self):
        from wxx import autograd

        return autograd.abs(self).forward(self)

    def sign(self):
        from wxx import autograd

        return autograd.sign(self).forward(self)

    def max(self, dim=None, keepdims=False, *, keepdim=None):
        from wxx import autograd

        if keepdim is not None:
            keepdims = keepdim
        return autograd.max(self).forward(self, dim, keepdims)

    def min(self, dim=None, keepdims=False, *, keepdim=None):
        from wxx import autograd

        if keepdim is not None:
            keepdims = keepdim
        return autograd.min(self).forward(self, dim, keepdims)

    def argmax(self, dim=None, keepdims=False, *, keepdim=None):
        from wxx import autograd

        if keepdim is not None:
            keepdims = keepdim
        return autograd.argmax(self).forward(self, dim, keepdims)

    def argmin(self, dim=None, keepdims=False, *, keepdim=None):
        from wxx import autograd

        if keepdim is not None:
            keepdims = keepdim
        return autograd.argmin(self).forward(self, dim, keepdims)

    # ------------------------------------------------------------------
    # FFT / complex
    # ------------------------------------------------------------------
    def rfft(self, n=None):
        from wxx import autograd

        return autograd.rfft(self).forward(self, n)

    def irfft(self, n=None):
        from wxx import autograd

        return autograd.irfft(self).forward(self, n)

    def complex_mul(self, other):
        from wxx import autograd

        return autograd.complex_mul(self, other).forward(self, other)

    def conj(self):
        from wxx import autograd

        return autograd.complex_conj(self).forward(self)


class Parameter(Tensor):
    """A tensor registered as a learnable parameter."""

    def __init__(self, tensor, requires_grad=True):
        super().__init__(
            shape=tensor.shape,
            buffer=tensor.buffer,
            offset=tensor.offset,
            stride=tensor.stride,
            requires_grad=requires_grad,
            dtype=tensor.type,
        )
        self._base = tensor
        self.Tensor = tensor
