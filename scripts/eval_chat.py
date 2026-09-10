"""
端到端聊天评测 (客服) — 需要后端在跑 + LLM 可用

用法:
    python scripts/eval_chat.py --eval scripts/eval_qa_customer.jsonl --base http://localhost:8001

输出: 每条问题的回答耗时/意图 + 成功率/平均延迟统计。
"""

import os
import sys
import json
import time
import argparse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description="端到端聊天评测 (客服)")
    parser.add_argument("--eval", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_qa_customer.jsonl"))
    parser.add_argument("--base", default="http://localhost:8001")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = []
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    ok, total, lats = 0, 0, []
    for r in rows:
        body = json.dumps({"query": r["question"], "session_id": f"eval_{r['id']}"}).encode("utf-8")
        req = urllib.request.Request(
            args.base + "/api/v1/chat/sync", data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            lat = time.time() - t0
            lats.append(lat)
            total += 1
            answer = data.get("answer", "")
            if answer.strip():
                ok += 1
            print(f"[{total}] {r['question'][:30]:<32} -> {lat:.1f}s | intent={data.get('_intent')} | handoff={data.get('handoff')}")
        except Exception as e:
            print(f"[x] {r['question'][:30]} -> 失败: {str(e)[:80]}")

    avg = sum(lats) / max(len(lats), 1)
    print(f"\n成功率 {ok}/{total} = {ok / total * 100:.1f}% | 平均延迟 {avg:.1f}s")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
