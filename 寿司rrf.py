def rrf_fusion(dense_result,sparse_result,top_k=5,k=60):
    rrf_score = {}
    seen = {}

    for rank,doc in enumerate(dense_result):
        cid =doc.get('chunk_id',doc.get('content',"")[:80])
        rrf_score[cid] = rrf_score.get(cid,0)+1.0/(k+rank+1)
        seen[cid] = doc

    for rank,doc in enumerate(sparse_result):
        cid =doc.get('chunk_id',doc.get('content',"")[:80])
        rrf_score[cid] = rrf_score.get(cid,0)+1.0/(k+rank+1)
        if cid not in seen:
            seen[cid] = doc

    sorted_cids = sorted(rrf_score.keys(),key=lambda c:rrf_score[c],reverse=True)[:top_k]

    result = []
    for cid in sorted_cids:
        doc = dict(seen[cid])
        doc["rrf_score"] = rrf_score[cid]
        result.append(doc)

    return result

from collections import defaultdict
from typing import List

def rrf_fusion2(dense_result,sparse_result,top_k=5,k=60)->List[dict]:
    rrf_score2 =defaultdict(float)
    docs = {}
    def get_cid(doc:dict) -> str:
        cid =doc.get('chunk_id')
        if cid:
            return str(cid)
        return doc.get('content',"") or repr(doc)

    def add_results(results:List[dict],source:str)->None:
        for rank,doc in enumerate(results):
            cid =get_cid(doc)
            rrf_score2[cid] +=1.0/(k+rank+1)

            if cid not in docs:
                docs[cid] = dict(doc)
                docs[cid]["source_scores"] = {}

            docs[cid]["source_scores"][source] = {
                "rank": rank + 1,
                "score": doc.get("score"),
            }
    add_results(dense_result,"dense")
    add_results(sparse_result,"sparse")