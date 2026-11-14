import subprocess, sys, time, os, json

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(path))[0]
cmd = [sys.executable, "src/main.py", path]
t0 = time.time()
p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
dt = time.time() - t0
rec = {
    "label": label, "command": " ".join(cmd), "file": path,
    "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
    "elapsed_s": round(dt, 2), "returncode": p.returncode,
    "stdout": p.stdout, "stderr": p.stderr,
}
with open(os.path.join("results", "batch_log.jsonl"), "a", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
with open(os.path.join("results", f"log_{label}.txt"), "w", encoding="utf-8") as f:
    f.write(p.stdout + "\n--- STDERR ---\n" + p.stderr + f"\n--- RC={p.returncode} time={dt:.2f}s ---\n")
print(p.stdout)
print("STDERR:", p.stderr)
print("RC", p.returncode, "time", round(dt, 2))
