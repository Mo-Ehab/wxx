import sys
import weakref

import wgpu

from .bindgroup import install_fast_bind_groups


class WebGPUContext:
    """Global manager holding the GPU device and compiled pipeline cache."""
    def __init__(self):
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync()
        self.pipeline_cache = {}
        # Bind-group creation dominates per-dispatch CPU cost; swap in the
        # cached-struct implementation when wgpu-py allows it.
        self.fast_bind_groups = install_fast_bind_groups(self.device)

    def get_pipeline(self, name: str, wgsl_code: str) -> wgpu.GPUComputePipeline:
        """Retrieves or compiles compute pipelines lazily."""
        if name not in self.pipeline_cache:
            module = self.device.create_shader_module(code=wgsl_code)
            pipeline = self.device.create_compute_pipeline(
                layout="auto",
                compute={"module": module, "entry_point": "main"}
            )
            self.pipeline_cache[name] = pipeline
        return self.pipeline_cache[name]


class GPUBufferPool:
    """Reuses completed GPU buffers to reduce allocation churn."""
    def __init__(self, device, max_cached_bytes=4 * 1024 * 1024):
        self.device = device
        self.max_cached_bytes = max_cached_bytes
        self._buffers = {}
        self._cached_ids = set()
        self._cached_bytes = 0
        self._active_buffers = {}

    def _take_cached(self, size, usage, label):
        key = (size, usage, label)
        buffers = self._buffers.get(key)
        while buffers:
            self._cached_bytes -= size
            buffer = buffers.pop()
            self._cached_ids.remove(id(buffer))
            if not buffers:
                del self._buffers[key]
            return buffer
        return None

    def create_buffer(self, size, usage):
        size = max(4, (size + 3) & ~3)
        usage = int(usage | wgpu.BufferUsage.COPY_DST)
        label = sys._getframe(1).f_code.co_name
        buffer = self._take_cached(size, usage, label)
        if buffer is None:
            buffer = self.device.create_buffer(label=label, size=size, usage=usage)
        self._track_active(buffer)
        return buffer

    def create_buffer_with_data(self, data, usage, label=None):
        data = memoryview(data)
        if data.nbytes % 4:
            raise ValueError("GPU buffer data size must be a multiple of 4 bytes")
        size = max(4, data.nbytes)
        usage = int(usage | wgpu.BufferUsage.COPY_DST)
        caller = sys._getframe(1).f_code.co_name
        label = caller if label is None else f"{caller}: {label}"
        buffer = self._take_cached(size, usage, label)
        if buffer is None:
            if not data.nbytes:
                buffer = self.device.create_buffer(label=label, size=size, usage=usage)
            else:
                buffer = self.device.create_buffer_with_data(
                    label=label, data=data, usage=usage
                )
        elif data.nbytes:
            self.device.queue.write_buffer(buffer, 0, data)
        self._track_active(buffer)
        return buffer

    def _track_active(self, buffer):
        self._active_buffers[id(buffer)] = (
            weakref.ref(buffer),
            buffer.size,
            buffer.label,
        )

    def release(self, buffer):
        buffer_id = id(buffer)
        if buffer_id in self._cached_ids:
            return
        self._active_buffers.pop(buffer_id, None)
        if getattr(buffer, "_internal", None) is None:
            return
        # Buffers flagged ``_no_cache`` are destroyed instead of cached
        # (used for large one-off scratch buffers such as FFT temporaries).
        if getattr(buffer, "_no_cache", False):
            try:
                buffer.destroy()
            except Exception:
                pass
            return
        size = buffer.size
        key = (size, int(buffer.usage), buffer.label)
        if size > self.max_cached_bytes or self._cached_bytes + size > self.max_cached_bytes:
            buffer.destroy()
            return
        self._buffers.setdefault(key, []).append(buffer)
        self._cached_ids.add(buffer_id)
        self._cached_bytes += size

    def get_memory_snapshot(self):
        active = []
        for buffer_id, (buffer_ref, size, label) in list(self._active_buffers.items()):
            if buffer_ref() is None:
                del self._active_buffers[buffer_id]
            else:
                active.append({"label": label, "size": size})

        cached = [
            {"label": buffer.label, "size": buffer.size}
            for buffers in self._buffers.values()
            for buffer in buffers
        ]
        return {
            "active": active,
            "cached": cached,
            "active_bytes": sum(buffer["size"] for buffer in active),
            "cached_bytes": self._cached_bytes,
        }

    def clear_cache(self):
        for buffers in self._buffers.values():
            for buffer in buffers:
                buffer.destroy()
        self._buffers.clear()
        self._cached_ids.clear()
        self._cached_bytes = 0


class _BatchedComputePass:
    def __init__(self, compute_pass):
        self._compute_pass = compute_pass

    def __getattr__(self, name):
        return getattr(self._compute_pass, name)

    def end(self):
        # The command buffer closes the shared pass when it is submitted.
        pass


class GPUCommandBuffer:
    """
    Batches GPU work into a single command encoder.

    Design:
      - get_encoder() lazily creates one shared encoder per batch.
      - Compute operations share one pass and encoder per batch.
      - Temporary buffers are kept alive through submission and readback.
      - read_buffer(), .item(), and __repr__() provide synchronous readback
        points where completed resources can be released.
    """
    def __init__(self, device, buffer_pool):
        self.device = device
        self.buffer_pool = buffer_pool
        self._encoder = None
        self._compute_pass = None
        self._batched_compute_pass = None
        self._alive_until_flush = []   # meta buffers retained until readback completes
        self._destroy_after_flush = []
        self._submitted_resources = []
        self._has_pending_work = False

    def get_encoder(self):
        """Returns the shared encoder, creating it lazily."""
        if self._encoder is None:
            self._encoder = self.device.create_command_encoder()
        return self._encoder

    def begin_compute_pass(self):
        """Returns a shared compute pass, opened lazily for the current batch."""
        if self._compute_pass is None:
            self._compute_pass = self.get_encoder().begin_compute_pass()
            self._batched_compute_pass = _BatchedComputePass(self._compute_pass)
        return self._batched_compute_pass

    def keep_alive(self, buf):
        """
        Retain a temporary buffer (e.g. metadata) until a readback completes.
        """
        self._alive_until_flush.append(buf)

    def defer_destroy(self, buf):
        """Destroy a buffer only after commands in the current batch finish."""
        if (
            self._encoder is None
            and not self._submitted_resources
            and not self._has_pending_work
        ):
            self.buffer_pool.release(buf)
        else:
            self._destroy_after_flush.append(buf)

    def flush(self, wait=False):
        """Submit pending work, optionally waiting before releasing completed resources."""
        if self._encoder is not None:
            if self._compute_pass is not None:
                self._compute_pass.end()
                self._compute_pass = None
                self._batched_compute_pass = None
            self.device.queue.submit([self._encoder.finish()])
            self._encoder = None
            self._has_pending_work = True
        self._submitted_resources.extend(self._alive_until_flush)
        self._submitted_resources.extend(self._destroy_after_flush)
        self._alive_until_flush.clear()
        self._destroy_after_flush.clear()
        if wait and self._has_pending_work:
            self._wait_for_queue()
            self.complete_pending()

    def _wait_for_queue(self):
        wait_source = self.device.create_buffer_with_data(
            data=bytes(4),
            usage=wgpu.BufferUsage.COPY_SRC,
            label="GPUCommandBuffer._wait_for_queue wait source",
        )
        try:
            wait_readback = self.device.create_buffer(
                label="GPUCommandBuffer._wait_for_queue readback",
                size=4,
                usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
            )
            try:
                encoder = self.device.create_command_encoder()
                encoder.copy_buffer_to_buffer(wait_source, 0, wait_readback, 0, 4)
                self.device.queue.submit([encoder.finish()])
                wait_readback.map_sync(wgpu.MapMode.READ)
                wait_readback.unmap()
            finally:
                wait_readback.destroy()
        finally:
            wait_source.destroy()

    def complete_pending(self):
        """Release resources after a synchronous readback confirms queue completion."""
        for buf in self._submitted_resources + self._destroy_after_flush:
            self.buffer_pool.release(buf)
        self._submitted_resources.clear()
        self._destroy_after_flush.clear()
        self._has_pending_work = False


gpu_ctx = WebGPUContext()
buffer_pool = GPUBufferPool(gpu_ctx.device)

def get_device():
    return gpu_ctx.device

def get_ctx():
    return gpu_ctx

cmd_buf = GPUCommandBuffer(get_device(), buffer_pool)

def get_cmd_buffer():
    return cmd_buf

def get_buffer_pool():
    return buffer_pool
