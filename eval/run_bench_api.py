import argparse
import json
import re
import urllib.request

SYS = "Sen foydali yordamchisan. Faqat bitta harf bilan javob ber (A, B, C yoki D)."


def load_items(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_choice(text):
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else None


def ask(base_url, question, max_tokens, timeout):
    payload = {
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": question + "\n\nJavobni faqat bitta harf bilan yoz:"},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--bench", default="eval/uzbench_mini.jsonl")
    ap.add_argument("--out", default="logs/results_falcon7b_api.json")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    items = load_items(args.bench)
    results = []
    correct = 0
    cat_stats = {}
    for it in items:
        try:
            gen = ask(args.url, it["question"], args.max_tokens, args.timeout).strip()
        except Exception as e:
            gen = f"XATO: {e}"
        pred = extract_choice(gen)
        ok = pred == it["answer"]
        correct += int(ok)
        cat = it.get("category", "boshqa")
        cat_stats.setdefault(cat, [0, 0])
        cat_stats[cat][0] += int(ok)
        cat_stats[cat][1] += 1
        results.append({**it, "pred": pred, "gen": gen[:80], "ok": ok})
        print(f"{it['id']:12s} kutilgan={it['answer']} model={pred} {'OK' if ok else 'X'} | {gen[:50]!r}", flush=True)

    n = len(items)
    summary = {
        "model": "Falcon3-7B-Instruct-1.58bit (server)",
        "url": args.url,
        "accuracy": round(correct / n, 4),
        "n": n,
        "per_category": {
            k: {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(cat_stats.items())
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
