import os
from importlib import resources
from typing import Dict

SHADER_PACKAGE = "wxx.shaders"

def _get_shaders_root():
    """Returns the Traversable root node for the shaders package."""
    return resources.files(SHADER_PACKAGE)

def walk_shader_files(traversable_dir=None, current_rel_path="") -> Dict[str, str]:
    """Recursively collects all .wgsl files into a dict keyed by relative path."""
    if traversable_dir is None:
        traversable_dir = _get_shaders_root()

    shader_files = {}
    
    if not traversable_dir.exists():
        return shader_files

    for item in traversable_dir.iterdir():
        rel_path = f"{current_rel_path}/{item.name}" if current_rel_path else item.name
        
        if item.is_dir():
            shader_files.update(walk_shader_files(item, rel_path))
        elif item.name.endswith(".wgsl") and item.is_file():
            shader_files[rel_path] = item.read_text(encoding="utf-8")

    return shader_files