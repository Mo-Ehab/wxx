import sys, os, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import wxx

PACKAGE_DIR = os.path.dirname(os.path.abspath(wxx.__file__))
SHADER_DIR = os.path.join(PACKAGE_DIR, "shaders")


def test_all_referenced_shaders_exist():
    assert os.path.isdir(SHADER_DIR), f"shader directory not found: {SHADER_DIR}"

    missing = []
    pattern = re.compile(r'shaders\["([^"]+)"\]')
    checked = 0
    for root, _, files in os.walk(PACKAGE_DIR):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            text = open(path, encoding="utf-8").read()
            for match in pattern.finditer(text):
                rel = match.group(1)
                checked += 1
                full = os.path.join(SHADER_DIR, *rel.split("/"))
                if not os.path.exists(full):
                    missing.append((os.path.relpath(path), rel))

    assert checked, "no shader references found; the lint is not scanning the package"
    assert not missing, f"missing shaders: {missing}"


TESTS = [("all_referenced_shaders_exist", test_all_referenced_shaders_exist)]

if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
