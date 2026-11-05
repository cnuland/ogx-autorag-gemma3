"""
OGX + AutoRAG + Gemma 3 27B — corrected working notebook.

This is the original notebook with four bugs fixed. Each fix is marked
`# FIX n:` with an explanation of what the original did and why it failed.

Prerequisites on the serving side (see README.md -- without these the
file_search calls either 400 or silently return ungrounded answers):

    vLLM:  --enable-auto-tool-choice
           --tool-call-parser=pythonic
           --chat-template=/tpl/gemma3-tools.jinja

    OGX:   tool_runtime provider  inline::file-search
           tool_groups            builtin::rag -> file-search
           vector_stores          default_provider_id + default_embedding_model
"""

import os
import shutil
import pprint

from ogx_client import OgxClient

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get(
    "OGX_BASE_URL",
    "http://ogx-service.<namespace>.svc.cluster.local:8321",
)
client = OgxClient(base_url=BASE_URL)

print("models   :", [m.id for m in client.models.list()])
print("providers:", [p.provider_id for p in client.providers.list()])

# ---------------------------------------------------------------------------
# Variables
#
# Model ids are "<provider_id>/<model_id>" as registered in the OGX config's
# registered_resources block -- not the raw HuggingFace repo name.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384                       # FIX 1: int, not the string '384'
INFERENCE_MODEL = "vllm-inference/gemma-quantized-w8a8"

SOURCE_FOLDER = "New_Documents"
STORED_FOLDER = "Stored"

# ---------------------------------------------------------------------------
# 1. Create the vector store
# ---------------------------------------------------------------------------
vector_store = client.vector_stores.create(
    name="operations",
    extra_body={
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIM,
        "provider_id": "pgvector",
    },
)
vector_store_id = vector_store.id
print(f"created vector store: {vector_store_id}")

# FIX 2 (the important one):
#   The original reassigned `vector_store_id = 'vs_1234'` right here, before
#   the upload loop. Every file was then attached to a store id that did not
#   exist, and the real store stayed empty -- so file_search found nothing.
#   Do NOT hardcode the id. Carry the value returned by create() through.

# ---------------------------------------------------------------------------
# 2. Upload and index documents
# ---------------------------------------------------------------------------
os.makedirs(STORED_FOLDER, exist_ok=True)

for filename in os.listdir(SOURCE_FOLDER):
    file_path = os.path.join(SOURCE_FOLDER, filename)
    if not os.path.isfile(file_path):
        continue

    if os.path.getsize(file_path) == 0:
        print(f"Warning: '{filename}' is 0 bytes. Skipping.")
        continue

    print(f"Uploading '{filename}'...")
    with open(file_path, "rb") as f:
        file_info = client.files.create(file=(filename, f), purpose="assistants")

    vector_store_file = client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=file_info.id,
        chunking_strategy={
            "type": "static",
            "static": {
                "max_chunk_size_tokens": 2000,
                "chunk_overlap_tokens": 200,
            },
        },
    )
    print(f"  indexed -> {vector_store_file.id} (status={vector_store_file.status})")

    shutil.move(file_path, os.path.join(STORED_FOLDER, filename))

print("Batch processing complete.")

counts = client.vector_stores.retrieve(vector_store_id=vector_store_id).file_counts
print("file_counts:", counts)       # completed should be > 0 before querying

# ---------------------------------------------------------------------------
# 3. Sanity-check retrieval BEFORE involving the model
#
# This bypasses the LLM entirely. If it returns content, your index and
# embeddings are fine and any bad answers are a tool-calling problem, not a
# RAG problem. This one call saves a lot of misdirected debugging.
# ---------------------------------------------------------------------------
probe = client.vector_stores.search(
    vector_store_id=vector_store_id,
    query="information trust barrier",
    max_num_results=3,
)
print(f"retrieval probe -> {len(probe.data)} results")
for i, d in enumerate(probe.data):
    text = " ".join(c.text for c in d.content) if d.content else "<no content>"
    print(f"  [{i}] score={getattr(d, 'score', None)} {text[:160]}")

# ---------------------------------------------------------------------------
# 4. Query with file_search
# ---------------------------------------------------------------------------
system_instructions = (
    "Answer using ONLY the retrieved documents. "
    "If the answer is not present, say you do not know. Cite the source."
)
prompt = "What is the information trust barrier?"

response = client.responses.create(
    model=INFERENCE_MODEL,            # FIX 3: was a llama model id while
                                      # INFERENCE_MODEL was defined and unused
    input=prompt,
    instructions=system_instructions,
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            # FIX 4: parallel_tool_calls was nested inside this dict and
            # passed as the string 'true'. It is a TOP-LEVEL boolean.
        }
    ],
    parallel_tool_calls=True,
)

pprint.pprint(response)
print(response.output_text)

# ---------------------------------------------------------------------------
# 5. Confirm retrieval actually happened
#
# A response can look plausible while never calling the tool -- the model
# answers from its priors and may invent citations. Always check that a
# file_search_call is present before trusting the answer.
# ---------------------------------------------------------------------------
used_file_search = any(
    getattr(item, "type", None) == "file_search_call" for item in response.output
)
print("file_search invoked:", used_file_search)

if not used_file_search:
    print(
        "WARNING: the model answered without retrieving. Check that vLLM was "
        "started with --enable-auto-tool-choice, --tool-call-parser=pythonic "
        "and --chat-template=/tpl/gemma3-tools.jinja. See README.md."
    )
