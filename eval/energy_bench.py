import argparse
import json
import time
import urllib.request
from pathlib import Path

RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")


def read_rapl():
    try:
        return int(RAPL.read_text())
    except (PermissionError, FileNotFoundError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompt", default="O'zbekiston haqida qisqa matn yoz.")
    ap.add_argument("--out", default="logs/energy.json")
    args = ap.parse_args()

    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.tokens,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    e0 = read_rapl()
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.load(resp)
    dt = time.time() - t0
    e1 = read_rapl()

    n = data.get("usage", {}).get("completion_tokens", args.tokens)
    result = {
        "url": args.url,
        "tokens": n,
        "seconds": round(dt, 2),
        "tokens_per_second": round(n / dt, 2),
    }
    if e0 is not None and e1 is not None:
        joules = (e1 - e0) / 1e6
        result["energy_j"] = round(joules, 2)
        result["energy_j_per_token"] = round(joules / n, 4)
    else:
        result["energy_j"] = None
        result["note"] = "RAPL o'qish uchun ruxsat yo'q: sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj"

    print(json.dumps(result, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
