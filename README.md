# OGX + AutoRAG + Gemma 3 27B — working configuration

Getting `file_search` RAG working end to end with Gemma 3 27B INT8 on
OGX (OpenShift AI 3.x), with PostgreSQL/pgvector as the vector store.

Everything here was reproduced on a live cluster: OGX server, pgvector,
Gemma 3 27B INT8 served by vLLM, a PDF indexed, and the RAG path exercised
and measured before and after the fix.

---

## TL;DR — the problem has two layers

`file_search` fails in two stages, and **fixing the first exposes a worse second one**.

| Stage | Symptom | Cause |
|---|---|---|
| 1 | `HTTP 400 "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` | vLLM not started with tool-calling flags |
| 2 | **No error. Model fabricates answers and citations.** | Gemma 3's chat template has no `tools` block, so the tool schemas never reach the model |

If you only add the flags, the error disappears and the system gets *worse* —
confidently wrong answers with invented sources. Both layers must be fixed.

---

## What actually goes wrong

### OGX does not fully broker `file_search`

`file_search` looks like a hosted/server-side tool, and OGX does register an
`inline::file-search` tool_runtime provider. But OGX converts it into a
`tool_choice: "auto"` function call that **the model must emit**. So the model's
tool-calling ability matters — which is the opposite of what the provider name
suggests.

### Gemma 3's chat template has no tool support

Verified directly against the tokenizer config:

```
mentions 'tools'      : False
mentions 'tool_call'  : False
mentions 'function'   : False
mentions 'tool_calls' : False
```

Tools passed in the request are **silently dropped** when the prompt is built.
No `--tool-call-parser` value fixes this — the parser only reads model *output*;
it cannot tell the model that tools exist.

Tested against vLLM directly, with OGX removed from the path:

```
BEFORE (stock template):
  tool_calls : None
  content    : "Okay, I've searched through the available documentation.
                The author is **<invented name>**... [invented URLs]"

AFTER (tool-aware template + matching parser):
  finish_reason: tool_calls
  tool_calls   : [{"name":"file_search","arguments":"{\"query\":\"<model-generated query>\"}"}]
  content      : ''
```

The correct answer was returned by retrieval at rank 2. The "before"
answer is entirely fabricated — including the claim to have searched.

### Template format and parser format must match

The subtle one. Gemma's *natural* output is a fenced block:

````
I need to search the files to answer this question.
```tool_code
[file_search(query="<model-generated query>")]
```
````

`pythonic` expects a **bare** literal — `[file_search(query="...")]` — with no
fence and no preamble. The fenced form parses as ordinary prose, so vLLM returns
`tool_calls: null` and the API response is **indistinguishable from "the model
chose not to search."** `functiongemma` does not parse that shape either (tested).

The template must force bare-pythonic output. Template and parser are a matched
pair: change one without the other and you get silent failure, never a loud one.

---

## The fix

### 1. vLLM flags

```yaml
args:
  - --enable-auto-tool-choice
  - --tool-call-parser=pythonic
  - --chat-template=/tpl/gemma3-tools.jinja
```

### 2. Mount the tool-aware chat template

`gemma3-tools.jinja` injects the function schemas into the prompt and mandates
the exact output format `pythonic` expects.

```bash
oc create configmap gemma3-chat-template --from-file=gemma3-tools.jinja
```

### 3. OGX config essentials

```yaml
providers:
  tool_runtime:
    - provider_id: file-search
      provider_type: inline::file-search

registered_resources:
  tool_groups:
    - toolgroup_id: builtin::rag
      provider_id: file-search

vector_stores:
  default_provider_id: pgvector
  default_embedding_model:
    provider_id: sentence-transformers
    model_id: all-MiniLM-L6-v2
    embedding_dimensions: 384
```

---

## Deploy

```bash
# 1. Postgres + pgvector
oc apply -f postgres-pgvector.yaml

# 2. pgvector extension MUST be created by a superuser (see gotcha 3)
oc exec deploy/ogx-postgres -- bash -c \
  'PGPASSWORD=<admin-pw> psql -U postgres -d ogx \
     -c "CREATE EXTENSION IF NOT EXISTS vector;" \
     -c "GRANT ALL ON SCHEMA public TO ogx;"'

# 3. Chat template + model server
oc create configmap gemma3-chat-template --from-file=gemma3-tools.jinja
oc apply -f gemma3-27b-deployment.yaml

# 4. OGX  (the label is REQUIRED -- see gotcha 1)
oc create configmap ogx-override-config --from-file=config.yaml=ogx-config.yaml
oc label configmap ogx-override-config ogx.io/watch=true
oc apply -f ogxserver.yaml

# 5. Confirm
oc get ogxserver          # PHASE should be Ready
curl -s http://ogx-service:8321/v1/health
curl -s http://ogx-service:8321/v1/models
```

---

## Gotchas (all of these cost real debugging time)

**1. ConfigMap needs `ogx.io/watch: "true"`**
Without the label the operator's informer never sees it:
`failed to reconcile override ConfigMap: failed to find referenced ConfigMap`
— while the ConfigMap plainly exists.

**2. vLLM provider key is `base_url`, not `url`**
Using `url` gives `ValueError: You must provide a URL in config.yaml`, which
reads like the field is missing rather than misnamed.

**3. pgvector extension must be created by a superuser**
The app user gets `permission denied to create extension "vector"`, surfaced as
the much vaguer `RuntimeError: Could not connect to PGVector database server`.

**4. Config structure**
`models:` and `tool_groups:` live under `registered_resources:`, and the
`storage:` block defining `kv_default` / `sql_default` is mandatory — other
sections reference those backend names.

**5. Provider `max_tokens` is max OUTPUT tokens**
Setting it to the full context window leaves no room for input:
```
This model's maximum context length is 32768 tokens. However, you requested
32768 output tokens and your prompt contains 82 characters (more than 0
characters, which is the upper bound for 0 input tokens).
```
Use ~4096.

---

## Bugs in the original notebook

See `autorag_notebook.py` for the corrected version with each fix marked.

**1. The vector store id was overwritten** (this alone breaks RAG):
```python
vector_store_id = vector_store.id      # real id
...
vector_store_id = 'vs_1234'            # <-- CLOBBERED before the upload loop
```
Files were attached to a store that did not exist; the real store stayed empty.

**2. Wrong model in the call** — `model='vllm-inference/meta-llamallama-32-3b-instruct'`
while `inference_model='vllm-inference/gemma-quantized-w8a8'` was defined and unused.

**3. `parallel_tool_calls` nested inside the tool dict** — it is a top-level
parameter, and a boolean, not the string `'true'`.

**4. `embedding_dimension` passed as the string `'384'`** rather than an int.

---

## Evaluating the RAG pipeline

`rag-eval.py` measures the two halves separately, which is the key to diagnosing
this class of problem.

### Setup

```bash
pip install openai
export OGX_BASE_URL=http://ogx-service.<namespace>.svc.cluster.local:8321/v1
python rag-eval.py
```

Edit the `EVAL` list at the top — each entry is a question plus the terms a
correct, grounded answer must contain:

```python
EVAL = [
 {"q": "<your question 1>", "must": ["term-a", "term-b"]},
 {"q": "<your question 2>", "must": ["term-c"]},
]
```

Keep the `must` terms distinctive enough that the model cannot produce them
from its priors — otherwise `groundedness_rate` will overstate how well
retrieval is working.

### What it measures

**Retrieval metrics** — no LLM involved, queries the vector store directly:

| Metric | Meaning |
|---|---|
| `hit_rate` | fraction of questions where an expected term appears in the top-k chunks |
| `mrr` | mean reciprocal rank — how high the correct chunk ranks |
| `p50_latency_s` | median vector search latency |

**Generation metrics** — full `responses.create` with `file_search`:

| Metric | Meaning |
|---|---|
| `file_search_invocation_rate` | **fraction of answers where the model actually called the tool** |
| `groundedness_rate` | fraction of answers containing the expected terms |
| `p50 / p95 latency` | end-to-end answer latency |

`file_search_invocation_rate` is the metric that exposes the silent failure.
An answer can read perfectly while never retrieving anything.

### How to read the output

**Run it twice** — once before applying the chat-template fix and once after.
The comparison is what makes the numbers meaningful.

- **Retrieval metrics barely move between runs.** Indexing and embeddings are
  independent of tool calling, so if `hit_rate` is already high while answers
  are poor, the problem is the tool-calling handshake, not your RAG. Running
  `vector_stores.search()` on its own separates those two in a single call, and
  is the first thing to check when answers look wrong.

- **`file_search_invocation_rate` is the number to watch.** It should move from
  zero to non-zero once the template and parser are aligned. If it stays at
  zero, the model is still never calling the tool — recheck the three vLLM
  flags and that your template's output format matches the parser.

- **Expect it to land below 1.0 with Gemma 3.** Some questions will still be
  answered from the model's priors without retrieving. That residual is the
  cost of bolting tool use onto a model never trained for it: the template
  teaches the call *format*, not the *judgement* of when to call. Measure it on
  your own corpus and decide whether the rate is acceptable for your use case.

- **`groundedness_rate` tracks invocation closely.** Where it lags, the model
  retrieved but did not use the context — usually a prompt/instruction issue
  rather than a plumbing one.

---

## Recommendation

The fix works, but if your measured `file_search_invocation_rate` is not close
to 1.0, a knowledge base that sometimes answers without retrieving may not be
acceptable. Options, in order of effort:

1. **Force retrieval.** Use a named `tool_choice` instead of `"auto"`.
   Grammar-constrained decoding guarantees the call, pushing invocation to 1.0
   at the cost of always retrieving. Best fit if every query should be grounded.

2. **Use a model with trained tool support.** Both of these have `tools` in
   their chat templates and first-class vLLM parsers, so none of the template
   work above is needed:
   - `RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8` — dense, INT8, parser `llama3_json`
   - `RedHatAI/granite-3.1-8b-instruct-quantized.w8a8` — dense, INT8, parser `granite`

3. **Keep Gemma 3 for non-tool workloads**, where it performed well.

---

## Files

| File | Purpose |
|---|---|
| `autorag_notebook.py` | **Corrected notebook** with all four bugs fixed and annotated |
| `gemma3-tools.jinja` | **The core fix** — tool-aware chat template |
| `ogx-config.yaml` | OGX providers config (goes in the watched ConfigMap) |
| `ogxserver.yaml` | OGXServer CR |
| `gemma3-27b-deployment.yaml` | vLLM serving Gemma 3 27B INT8 |
| `postgres-pgvector.yaml` | Postgres 15 + pgvector |
| `index-pdf.py` | Create a vector store and index a PDF |
| `rag-eval.py` | Retrieval + groundedness evaluation |
