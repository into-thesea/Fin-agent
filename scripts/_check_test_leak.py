"""测试集去泄漏检查: 计算测试集每条问题与 dev 集 + 提示词例句的 bigram 重叠度."""
import json
import re
from pathlib import Path

ROOT = Path(r"D:\aiproject")

def bigrams(text: str) -> set:
    """中文按字符 bigram, 去掉标点空白."""
    chars = re.sub(r"[^\u4e00-\u9fff a-zA-Z0-9]", "", text)
    return {chars[i:i+2] for i in range(len(chars)-1)} if len(chars) >= 2 else set()

# 1. 读 dev 集
dev_questions = []
with open(ROOT / "data/eval/finance_qa_golden.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            dev_questions.append(json.loads(line)["question"])

# 2. 读提示词, 提取所有例句 (引号里的句子)
prompt_text = (ROOT / "src/agents/router_agent.py").read_text(encoding="utf-8")
# 提取 INTENT_DETECTION_PROMPT 块
m = re.search(r'INTENT_DETECTION_PROMPT\s*=\s*"""(.*?)"""', prompt_text, re.DOTALL)
prompt_block = m.group(1) if m else ""
# 提取所有中文引号或英文引号里的例句
prompt_examples = re.findall(r'[""](.+?)[""]', prompt_block)
prompt_examples += re.findall(r'"([^"]{4,})"', prompt_block)

print(f"dev 集问题数: {len(dev_questions)}")
print(f"提示词例句数: {len(prompt_examples)}")

# 3. 合并所有"已见过"的语料
seen_corpus = dev_questions + prompt_examples
seen_bigrams = set()
for q in seen_corpus:
    seen_bigrams |= bigrams(q)

print(f"已见语料 bigram 总数: {len(seen_bigrams)}")

# 4. 逐条检查测试集
test_items = []
with open(ROOT / "data/eval/finance_qa_test.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            test_items.append(json.loads(line))

print(f"\n测试集共 {len(test_items)} 条, 泄漏检查结果:")
print("=" * 70)

high_risk = []
for item in test_items:
    q = item["question"]
    bg = bigrams(q)
    overlap = bg & seen_bigrams
    ratio = len(overlap) / len(bg) if bg else 0
    # 找最相似的 dev 问题
    best_match, best_ratio = "", 0
    for dq in dev_questions:
        dbg = bigrams(dq)
        r = len(bg & dbg) / len(bg | dbg) if (bg | dbg) else 0
        if r > best_ratio:
            best_ratio, best_match = r, dq
    flag = " *** 高风险 ***" if ratio > 0.5 else ""
    if ratio > 0.5:
        high_risk.append(item)
    print(f"{item['id']} overlap={ratio:.0%} jaccard_top1={best_ratio:.0%}{flag}")
    print(f"  Q: {q}")
    if best_ratio > 0.2:
        print(f"  最像dev: {best_match}")

print(f"\n高风险条目: {len(high_risk)}/{len(test_items)}")
if high_risk:
    for item in high_risk:
        print(f"  {item['id']}: {item['question']}")
else:
    print("全部通过, 无明显泄漏.")
