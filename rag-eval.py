"""RAG evaluation over the OGX/pgvector index (RAGAS-style metrics)."""
import os
from openai import OpenAI
import json, time, statistics as st

BASE = os.environ.get("OGX_BASE_URL",
        "http://ogx-service.<namespace>.svc.cluster.local:8321/v1")
client=OpenAI(base_url=BASE, api_key="none", timeout=300.0)
vsid=open("/tmp/vsid.txt").read().strip()
MODEL="vllm-inference/gemma-quantized-w8a8"

# Synthetic eval set grounded in "Managing the Information Trust Barrier"
EVAL=[
 {"q":"What is the information trust barrier about?","must":["trust","routing"]},
 {"q":"Who is the author of this presentation?","must":["nuland"]},
 {"q":"What is Praxis Core built with?","must":["rust","pingora","tokio","rustls"]},
 {"q":"Why is cost-optimized routing a data exfiltration path?","must":["routing","data"]},
 {"q":"What technologies are mentioned alongside llm-d?","must":["nemo","guardrails","praxis","nvidia"]},
 {"q":"What classification is this deck marked with?","must":["internal","nda","shared"]},
]

def retrieval_metrics():
    hits=0; rr=[]; lat=[]
    for e in EVAL:
        t0=time.time()
        r=client.vector_stores.search(vector_store_id=vsid, query=e["q"], max_num_results=5)
        lat.append(time.time()-t0)
        texts=[" ".join(c.text for c in d.content).lower() for d in r.data if d.content]
        rank=None
        for i,t in enumerate(texts):
            if any(m in t for m in e["must"]): rank=i+1; break
        if rank: hits+=1; rr.append(1.0/rank)
        else: rr.append(0.0)
        print(f"  [{'HIT ' if rank else 'MISS'}] rank={rank} q={e['q'][:48]}")
    return {"hit_rate":hits/len(EVAL), "mrr":sum(rr)/len(EVAL),
            "p50_latency_s":round(st.median(lat),3), "n":len(EVAL)}

def answer_metrics():
    res=[]
    for e in EVAL:
        t0=time.time()
        r=client.responses.create(model=MODEL,
            input=e["q"],
            instructions="Answer ONLY from the retrieved documents. If absent, say 'not in the documents'.",
            tools=[{"type":"file_search","vector_store_ids":[vsid]}])
        dt=time.time()-t0
        out=(r.output_text or "").lower()
        grounded=any(m in out for m in e["must"])
        used_tool=any(getattr(i,"type",None)=="file_search_call" for i in r.output)
        res.append({"q":e["q"],"grounded":grounded,"used_file_search":used_tool,
                    "latency_s":round(dt,2),"answer":(r.output_text or "")[:160]})
        print(f"  [{'OK  ' if grounded else 'FAIL'}] tool={used_tool} {dt:5.1f}s  {e['q'][:44]}")
    return res

print("="*64); print("RETRIEVAL METRICS"); print("="*64)
rm=retrieval_metrics(); print(json.dumps(rm,indent=2))
print(); print("="*64); print("ANSWER / GENERATION METRICS"); print("="*64)
am=answer_metrics()
g=sum(1 for a in am if a["grounded"]); t=sum(1 for a in am if a["used_file_search"])
summary={"groundedness_rate":round(g/len(am),3),"file_search_invocation_rate":round(t/len(am),3),
         "p50_latency_s":round(st.median([a['latency_s'] for a in am]),2),
         "p95_latency_s":round(sorted([a['latency_s'] for a in am])[-1],2)}
print(json.dumps(summary,indent=2))
json.dump({"retrieval":rm,"answers":am,"summary":summary}, open("/tmp/rag_eval.json","w"), indent=2)
print("\nwrote /tmp/rag_eval.json")
