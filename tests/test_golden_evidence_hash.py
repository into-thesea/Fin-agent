"""虚拟产品文档的 evidence 指纹必须落到真实渲染内容上

evidence 里 47 条写的是 prod_pNNN.md —— 磁盘上没有这个文件，产品文档已改成从
catalog.jsonl 渲染。按路径哈希只会拿到 MISSING，于是这批用例「内容变了没有」
**永远检测不出来**（静默失效），而且全堆进「缺失」桶，把 `--check` 的「已变化」
桶挤成 0 条 —— 比报错更糟，报错至少有人看见。

反向也要守：真不存在的名字必须还是 MISSING，哈希函数不能退化成「什么都能算出来」。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import golden_bind_version as g  # noqa: E402


def test_virtual_product_doc_resolves():
    h = g.evidence_hash("prod_p002.md")
    assert h != "MISSING" and len(h) == 16


def test_two_products_hash_differently():
    """按文档聚合指纹，不是把整个 catalog.jsonl 拍成一个哈希"""
    assert g.evidence_hash("prod_p001.md") != g.evidence_hash("prod_p002.md")


def test_stable_across_calls():
    assert g.evidence_hash("prod_p002.md") == g.evidence_hash("prod_p002.md")


def test_real_file_still_path_hashed():
    """磁盘上真有的 evidence 走原路径，不受虚拟文档分支影响"""
    rows = [json.loads(line) for line in open(g.GOLDEN, encoding="utf-8") if line.strip()]
    ev = next(e for r in rows for e in r.get("evidence", [])
              if os.path.exists(os.path.join(g.KB_DIR, e)))
    assert g.evidence_hash(ev) == g.get_file_hash(os.path.join(g.KB_DIR, ev))


def test_unknown_name_still_missing():
    assert g.evidence_hash("__no_such_doc__.md") == "MISSING"
