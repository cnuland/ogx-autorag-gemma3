import os
from openai import OpenAI
import time, json

BASE = os.environ.get("OGX_BASE_URL",
        "http://ogx-service.<namespace>.svc.cluster.local:8321/v1")
client = OpenAI(base_url=BASE, api_key="none")

print("=== create vector store ===")
vs = client.vector_stores.create(
    name="operations",
    extra_body={
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "provider_id": "pgvector",
    },
)
print("vector_store_id =", vs.id)

print("=== upload file ===")
with open("/tmp/trust-barrier.pdf", "rb") as f:
    fi = client.files.create(file=("trust-barrier.pdf", f), purpose="assistants")
print("file_id =", fi.id, "bytes=", fi.bytes)

print("=== attach to vector store ===")
vsf = client.vector_stores.files.create(
    vector_store_id=vs.id,
    file_id=fi.id,
    chunking_strategy={"type":"static","static":{"max_chunk_size_tokens":2000,"chunk_overlap_tokens":200}},
)
print("vs_file_id =", vsf.id, "status =", vsf.status)

for i in range(60):
    cur = client.vector_stores.files.retrieve(vector_store_id=vs.id, file_id=fi.id)
    print(f"  [{i}] status={cur.status} err={getattr(cur,'last_error',None)}")
    if cur.status in ("completed","failed","cancelled"): break
    time.sleep(5)

print("=== store counts ===")
print(client.vector_stores.retrieve(vector_store_id=vs.id))
with open("/tmp/vsid.txt","w") as f: f.write(vs.id)
