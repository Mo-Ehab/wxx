import sys, os, re, glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SHADER_DIR = os.path.join(os.path.dirname(__file__), "..", "wxx", "shaders")


def test_all_referenced_shaders_exist():
    missing = []
    pattern = re.compile(r'shaders\["([^"]+)"\]')
    for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), "..", "wxx")):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            text = open(path, encoding="utf-8").read()
            for m in pattern.finditer(text):
                rel = m.group(1)
                full = os.path.join(SHADER_DIR, *rel.split("/"))
                if not os.path.exists(full):
                    missing.append((os.path.relpath(path), rel))
    assert not missing, f"missing shaders: {missing}"


TESTS = [("all_referenced_shaders_exist", test_all_referenced_shaders_exist)]

if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
