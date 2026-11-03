import sys, os, subprocess, glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

files = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
# exclude run_all itself
files = [f for f in files if os.path.basename(f) != "run_all.py"]

overall_ok = True
for f in files:
    print("=" * 70)
    print("RUNNING", os.path.basename(f))
    print("=" * 70)
    r = subprocess.run([sys.executable, f], cwd=HERE, capture_output=False)
    if r.returncode != 0:
        overall_ok = False
        print(f"^^^ {os.path.basename(f)} FAILED (exit {r.returncode})")

print("\n" + "=" * 70)
print("ALL TEST FILES:", "PASS" if overall_ok else "FAIL")
sys.exit(0 if overall_ok else 1)
