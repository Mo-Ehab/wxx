
from __future__ import annotations

import wgpu


def install_fast_bind_groups(device):
    """Replace ``device.create_bind_group`` with a cached-struct version.

    Returns ``True`` when the fast path was installed.  Falls back silently on
    any incompatibility.
    """
    try:
        if getattr(device.create_bind_group, "_wxx_fast", False):
            return True  # already installed; do not wrap the wrapper
        from wgpu.backends.wgpu_native import _api

        ffi = _api.ffi
        libf = _api.libf
        new_struct_p = _api.new_struct_p
        to_c_string_view = _api.to_c_string_view
        whole_size = _api.lib.WGPU_WHOLE_SIZE
        GPUBindGroup = wgpu.GPUBindGroup

        empty_label = to_c_string_view("")
        arrays = {}
        descriptors = {}

        def _entry_array(count):
            array = arrays.get(count)
            if array is None:
                array = ffi.new("WGPUBindGroupEntry[]", count)
                arrays[count] = array
                descriptors[count] = new_struct_p(
                    "WGPUBindGroupDescriptor *",
                    nextInChain=ffi.NULL,
                    label=empty_label,
                    layout=ffi.NULL,
                    entryCount=count,
                    entries=array,
                )
            return array, descriptors[count]

        def create_bind_group(*, label="", layout, entries):
            try:
                count = len(entries)
                array, descriptor = _entry_array(count)
                for index, entry in enumerate(entries):
                    resource = entry["resource"]
                    buffer = resource["buffer"]
                    internal = buffer._internal
                    if internal is None:
                        raise TypeError("buffer has been released")
                    slot = array[index]
                    slot.binding = int(entry["binding"])
                    slot.buffer = internal
                    slot.offset = resource.get("offset", 0)
                    size = resource.get("size")
                    slot.size = whole_size if size is None else int(size)
                    slot.sampler = ffi.NULL
                    slot.textureView = ffi.NULL
                descriptor.layout = layout._internal
                internal_id = libf.wgpuDeviceCreateBindGroup(
                    device._internal, descriptor
                )
                return GPUBindGroup(label, internal_id, device)
            except (KeyError, TypeError, AttributeError):
                # Non-buffer resource, unusual descriptor, or a wgpu-py layout
                # we do not understand: use the reference implementation.
                return _original(label=label, layout=layout, entries=entries)

        create_bind_group._wxx_fast = True
        _original = device.create_bind_group
        device.create_bind_group = create_bind_group
        return True
    except Exception:  # pragma: no cover - depends on the wgpu-py build
        return False
