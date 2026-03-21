# Embedding Models for Semantic Index — Comprehensive Analysis

> This document compares embedding models suitable for code + documentation
> search, explains the default choice, and helps users pick the right model
> for their setup.

---

## Why BAAI/bge-m3 Is the Default

Before diving into the full comparison, here's the reasoning for the default choice.
The selection was driven by five hard constraints specific to this SKILL:

**1. Dual-mode compatibility (the killer constraint)**

The SKILL supports both OpenRouter (remote) and HuggingFace (local) providers.
The default model must work in both modes, producing the same vectors so users
can build an index with one provider and search with the other. BGE-M3 is
available on OpenRouter ($0.01/M tokens, 8K context) and locally via
sentence-transformers, making it the strongest dual-mode candidate among models
that also satisfy the other constraints below.

**2. Context window ≥ 2048 tokens**

Code functions can be long. A 512-token context window (MiniLM, BGE-Large, E5,
mxbai) means chunks get truncated, losing the end of functions or the bottom
half of documentation sections. BGE-M3's 8,192-token window handles even
large class definitions without truncation. This is table-stakes for code search.

**3. CPU-viable performance**

Not everyone has a GPU. The model must index a 500-file project in reasonable
time on a modern laptop CPU. At 568M parameters, BGE-M3 is slower than
lightweight models (~90-180 seconds for 500 files vs ~30-60 for Nomic v1.5's
137M), but still viable. Anything 7B+ is GPU-only.

**4. Apache 2.0 license**

The SKILL is designed to be freely distributable. The default model must be
usable without licensing concerns. This eliminates Jina v3 (proprietary),
mxbai-embed (proprietary), and some Qwen variants. BGE-M3 is Apache 2.0.

**5. Multilingual out of the box**

Real-world codebases often contain comments, documentation, and commit messages
in multiple languages. BGE-M3 was trained on 100+ languages with explicit
cross-lingual alignment. This future-proofs the default — it works for
English-only projects just as well, while also handling multilingual ones
without requiring a model switch.

**Bonus: No prefix required**

Unlike some models (Nomic v1.5 requires `search_document:` / `search_query:`
prefixes), BGE-M3 works without any prefix. This simplifies integration and
eliminates a common source of misconfiguration.

### What BGE-M3 Is NOT Best At

To be transparent about the tradeoffs:

- **Not the fastest on CPU**: At 568M parameters, it's ~2-3x slower than
  Nomic v1.5 (137M) or all-MiniLM-L6-v2 (22M). If you need sub-30-second
  indexing of large projects on CPU, consider nomic-embed-text-v1.5 as a
  local-only lightweight alternative.
- **Not the highest absolute quality**: 7-8B models (SFR-Code-7B, Qwen3-Embedding-8B)
  score 10-15% higher on code retrieval benchmarks. If you have a GPU and only
  use local mode, those are better.
- **Not code-specialized**: It's a general-purpose multilingual model. Models
  like SFR-Embedding-Code and Mistral Codestral Embed are trained specifically
  on code corpora and understand programming constructs better.
- **Larger download (~1.1 GB vs ~274 MB)**: First-time setup takes longer than
  lightweight models, but this is a one-time cost.
- **~1.5-2 GB RAM vs ~500 MB**: Still fits comfortably on any modern machine,
  but worth noting for very constrained environments.

### Lightweight Alternative: nomic-embed-text-v1.5 (Local Only)

If your project is English-only and speed matters more than multilingual
coverage, `nomic-ai/nomic-embed-text-v1.5` is an excellent local-only choice:
137M parameters, 8K context, ~500-750 sentences/sec on CPU, Apache 2.0,
Matryoshka dimension support (768 down to 64). Note: Nomic v1.5 is **not
available on OpenRouter** — it works only with the HuggingFace local provider.
It requires `search_document:` / `search_query:` task prefixes (handled
automatically by the SKILL).

---

## The Full Comparison

### Tier 1: Heavyweight (7B+ params, GPU required)

These deliver the best quality but require significant hardware.

#### NVIDIA Llama-Embed-Nemotron-8B

| Attribute | Value |
|-----------|-------|
| Parameters | 8B |
| Dimensions | 4096 |
| Context | 32,768 tokens |
| Download | ~16 GB |
| RAM/VRAM | ~16 GB VRAM (GPU required) |
| License | Open (Apache 2.0 based) |
| Prefixes | Instruction-based query formatting |
| sentence-transformers | Yes |
| OpenRouter | No |
| MTEB Rank | #1 Multilingual MTEB (Oct 2025) |

**Pros**: Highest multilingual benchmark scores, massive context window, excellent
code understanding from Llama 3.1 base.
**Cons**: 16 GB VRAM means high-end GPU only. Not on OpenRouter. Overkill for
most project-level code search. Slow on CPU (unusable).
**Best for**: Large enterprise codebases with multilingual documentation, when
GPU infrastructure is available.

#### NVIDIA Llama Nemotron Embed VL 1B V2

| Attribute | Value |
|-----------|-------|
| Parameters | 1B |
| Dimensions | — |
| Context | 131,072 tokens |
| Download | ~2 GB |
| RAM/VRAM | ~2-4 GB |
| License | Open |
| Prefixes | — |
| sentence-transformers | TBD |
| OpenRouter | **Yes (FREE)** |
| Special | Multimodal (text + image) |

**Pros**: FREE on OpenRouter, massive 131K context window, multimodal (can embed
images alongside text — useful for documentation with diagrams). 1B parameters
is borderline CPU-viable.
**Cons**: Newer model with less community track record. Multimodal capability
may not be needed for pure code/text search. Local HuggingFace support TBD.
**Best for**: Cost-conscious setups via OpenRouter, or projects with image-heavy
documentation (diagrams, screenshots, architecture charts).

#### Qwen3-Embedding-8B

| Attribute | Value |
|-----------|-------|
| Parameters | 8B |
| Dimensions | 4096 (configurable) |
| Context | 32,768 tokens |
| Download | ~16 GB |
| RAM/VRAM | ~32 GB recommended |
| License | Proprietary (Qwen family) |
| Prefixes | Instruction-based |
| sentence-transformers | Yes |
| OpenRouter | **Yes** ($0.01/M tokens) |
| MTEB Rank | #1 Multilingual MTEB (June 2025, score 70.58) |

**Pros**: Top MTEB scores, strong code retrieval, configurable output dimensions,
Flash Attention 2 support, **multilingual**, available on OpenRouter.
**Cons**: Proprietary license, massive memory footprint, GPU-only for local use.
**Best for**: Teams with GPU clusters who need maximum retrieval quality, or via
OpenRouter when cost ($0.01/M) matters less than quality.

#### Qwen3-Embedding-4B

| Attribute | Value |
|-----------|-------|
| Parameters | 4B |
| Dimensions | configurable |
| Context | 33,792 tokens |
| Download | ~8 GB |
| RAM/VRAM | ~16 GB recommended |
| License | Proprietary (Qwen family) |
| Prefixes | Instruction-based |
| sentence-transformers | Yes |
| OpenRouter | **Yes** ($0.02/M tokens) |

**Pros**: Same Qwen3 quality at half the parameters, 33K context, multilingual,
available on OpenRouter.
**Cons**: Proprietary license, still GPU-only for local use, 4B is heavy for CPU.
**Best for**: When Qwen3-8B is overkill but you want Qwen3 quality via OpenRouter.

#### Qwen3-Embedding-0.6B

| Attribute | Value |
|-----------|-------|
| Parameters | 0.6B |
| Dimensions | configurable |
| Context | 8,192 tokens |
| Download | ~1.2 GB |
| RAM/VRAM | ~2 GB |
| License | Proprietary (Qwen family) |
| Prefixes | Instruction-based |
| sentence-transformers | Yes |
| OpenRouter | **Yes** |

**Pros**: Lightweight Qwen3 variant, 8K context (good for code), multilingual,
available on both OpenRouter and HuggingFace (dual-mode viable), CPU-borderline.
**Cons**: Proprietary license, 0.6B is slower on CPU than Nomic v1.5 (137M) or
BGE-M3 (568M may actually be faster due to optimized architecture). Quality
below the 4B/8B siblings.
**Best for**: Dual-mode setup when you want Qwen3 multilingual quality with a
smaller footprint than BGE-M3.

#### Salesforce SFR-Embedding-Code (CodeXEmbed)

| Attribute | Value |
|-----------|-------|
| Parameters | 400M / 2B / 7B variants |
| Dimensions | Model-dependent |
| Context | 32,768 tokens |
| Download | 800MB / 4GB / 14GB |
| RAM/VRAM | 2-3GB / 6-8GB / 14-16GB |
| License | Open source |
| Prefixes | Instruction-based code query formatting |
| sentence-transformers | Yes |
| OpenRouter | No |
| Code Benchmark | #1 on CoIR benchmark, 20%+ over Voyage-Code |

**Pros**: Purpose-built for code, best code retrieval scores, 32K context,
three size variants so you can pick your tradeoff.
**Cons**: Not on OpenRouter (local only), the 2B+ variants need GPU, code-only
focus may underperform on pure documentation search.
**Best for**: Code-heavy projects where documentation search is secondary.
The 400M variant is viable on CPU.

---

### Tier 2: Mid-size (300M-600M params, CPU viable with patience)

Good balance of quality and resource requirements.

#### BAAI BGE-M3 (DEFAULT)

| Attribute | Value |
|-----------|-------|
| Parameters | 568M |
| Dimensions | 1024 |
| Context | 8,192 tokens |
| Download | ~1.1 GB |
| RAM/VRAM | ~1.5-2 GB |
| License | Apache 2.0 |
| Prefixes | Not required |
| sentence-transformers | Yes |
| OpenRouter | **Yes** ($0.01/M tokens, 8K context) |
| MTEB Score | ~63.0 average |

**Pros**: 100+ language support, hybrid retrieval (dense + sparse + multi-vector),
no prefix needed (simpler integration), 8K context, Apache 2.0, **dual-mode
compatible** (OpenRouter + HuggingFace), cheapest multilingual 8K option on
OpenRouter.
**Cons**: 568M parameters so ~2-3x slower on CPU than Nomic v1.5 (137M). Not
code-specialized. Multi-vector mode adds complexity.
**Best for**: **The default choice.** Multilingual, dual-mode, 8K context, no
prefixes, Apache 2.0, affordable on OpenRouter.

#### Alibaba GTE-Large-En-v1.5

| Attribute | Value |
|-----------|-------|
| Parameters | 434M |
| Dimensions | 1024 |
| Context | 8,192 tokens (HuggingFace) / 512 tokens (OpenRouter `thenlper/gte-large`) |
| Download | ~850 MB |
| RAM/VRAM | ~2 GB |
| License | Open source |
| Prefixes | Not required |
| sentence-transformers | Yes |
| OpenRouter | **Yes** (`thenlper/gte-large`, $0.01/M) but only 512 context |
| MTEB Score | 65.39 |

**Pros**: Strong retrieval scores, 8K context when used locally, no prefixes
needed, clean integration.
**Cons**: English-only. OpenRouter variant has only 512 context (a severe
limitation for code — use local HuggingFace for 8K context).
**Best for**: English-only projects where you prioritize simplicity (no prefixes)
and use the local provider for full 8K context.

#### Jina Embeddings v3

| Attribute | Value |
|-----------|-------|
| Parameters | 570M |
| Dimensions | 1024 (truncatable to 32+ via MRL) |
| Context | 8,192 tokens |
| Download | ~1.2 GB |
| RAM/VRAM | ~2 GB |
| License | Proprietary |
| Prefixes | Task-specific LoRA adapters |
| sentence-transformers | Yes |
| OpenRouter | **No** |
| MTEB Score | 64.44 (multilingual) |

**Pros**: Task-specific LoRA adapters (different embedding strategies for
retrieval vs. clustering vs. classification), MRL dimension reduction, good
multilingual support. Has its own API (api.jina.ai) as an alternative to
OpenRouter.
**Cons**: Proprietary license, not on OpenRouter, 570M is heavy for CPU,
LoRA adapter switching adds complexity, not code-specialized.
**Best for**: Multi-purpose embedding needs (not just search but also
clustering/classification). Requires Jina's own API for remote use.

#### Nomic Embed Text v2-MoE

| Attribute | Value |
|-----------|-------|
| Parameters | 475M total / 305M active (MoE) |
| Dimensions | 256-768 (Matryoshka) |
| Context | 512 tokens |
| Download | ~1.9 GB |
| RAM/VRAM | ~600MB-1 GB |
| License | Apache 2.0 |
| Prefixes | Required (same as v1.5) |
| sentence-transformers | Yes (trust_remote_code=True) |
| OpenRouter | No |
| Downloads | 1.4M (growing fast) |

**Pros**: Mixture-of-Experts architecture (only 305M active), multilingual
(100+ languages), outperforms models 2x its active size, Apache 2.0, fast
inference due to sparse routing.
**Cons**: **512 token context window** — a critical limitation for code
chunks that regularly exceed this. Newer model with less production track
record. Not on OpenRouter yet.
**Best for**: Multilingual projects with short code chunks. Would be an
excellent default once/if an 8K context variant is released.

#### mixedbread mxbai-embed-large-v1

| Attribute | Value |
|-----------|-------|
| Parameters | ~335M |
| Dimensions | 1024 (reducible to 512+ via MRL) |
| Context | 512 tokens |
| Download | ~700 MB |
| RAM/VRAM | ~2 GB |
| License | Proprietary |
| Prefixes | Not required |
| sentence-transformers | Yes |
| OpenRouter | **No** |
| MTEB Score | Competitive with BGE variants |

**Pros**: No prefix needed, binary quantization support (extreme compression),
MRL dimension reduction.
**Cons**: 512 token context (too short for code), proprietary license, not on
OpenRouter.
**Best for**: Text-heavy projects where storage efficiency matters more than
code search quality. Local only.

#### Mistral Codestral Embed 2505

| Attribute | Value |
|-----------|-------|
| Parameters | ~540M |
| Dimensions | — |
| Context | 8,192 tokens |
| Download | — (API-only) |
| RAM/VRAM | — (API-only) |
| License | Proprietary |
| Prefixes | Not required |
| sentence-transformers | No |
| OpenRouter | **Yes** ($0.15/M tokens) |

**Pros**: **Code-specialized** — designed specifically for embedding code
databases, repositories, and powering coding assistants. 8K context (good for
code). Available on OpenRouter.
**Cons**: Proprietary, expensive ($0.15/M — 15x BGE-M3), API-only (no local
HuggingFace option), not multilingual, no dual-mode.
**Best for**: Code-heavy projects where you want the best code-specific
embedding quality via API and cost is not a concern.

#### Google Gemini Embedding 001

| Attribute | Value |
|-----------|-------|
| Parameters | — (proprietary) |
| Dimensions | — |
| Context | 20,480 tokens |
| Download | — (API-only) |
| RAM/VRAM | — (API-only) |
| License | Proprietary |
| Prefixes | Not required |
| sentence-transformers | No |
| OpenRouter | **Yes** ($0.15/M tokens) |
| MTEB Rank | Top spot on MTEB Multilingual leaderboard |

**Pros**: Excellent quality across domains (science, legal, finance, coding),
20K context (largest among mid-tier models on OpenRouter), strong multilingual
support, top MTEB scores.
**Cons**: Proprietary, expensive ($0.15/M), API-only (no local option), no
dual-mode. Google-specific terms of service.
**Best for**: Quality-first setups where API cost is acceptable and you want
the longest context window available on OpenRouter.

#### Mistral Embed 2312

| Attribute | Value |
|-----------|-------|
| Parameters | — (proprietary) |
| Dimensions | 1024 |
| Context | 8,192 tokens |
| Download | — (API-only) |
| RAM/VRAM | — (API-only) |
| License | Proprietary |
| Prefixes | Not required |
| sentence-transformers | No |
| OpenRouter | **Yes** ($0.10/M tokens) |

**Pros**: Semantic search and RAG-optimized, 1024-dimensional vectors, 8K
context, established model.
**Cons**: Proprietary, API-only, older (late 2023), not multilingual, no
dual-mode, expensive relative to BGE-M3 ($0.10 vs $0.01).
**Best for**: Existing Mistral users. Otherwise BGE-M3 is better in every
dimension for this SKILL's use case.

---

### Tier 3: Lightweight (< 150M params, fast CPU inference)

Best for resource-constrained environments or when speed matters most.

#### nomic-ai/nomic-embed-text-v1.5

| Attribute | Value |
|-----------|-------|
| Parameters | 137M |
| Dimensions | 768 (64-768 via Matryoshka) |
| Context | 8,192 tokens |
| Download | ~274 MB |
| RAM/VRAM | ~500 MB |
| License | Apache 2.0 |
| Prefixes | Required: `search_document:` / `search_query:` |
| sentence-transformers | Yes (trust_remote_code=True) |
| OpenRouter | **No** |
| Downloads | 9.7M (most downloaded in family) |
| CPU Speed | ~500-750 sentences/sec |

**Pros**: Best quality-to-size ratio in its class, 8K context window (critical
for code), Matryoshka dimensions, Apache 2.0, very fast on CPU, tiny download,
proven in production (9.7M downloads).
**Cons**: Requires task prefixes, not code-specialized, English-only, **not on
OpenRouter** (HuggingFace local only), lower absolute quality than 335M+ models.
**Best for**: Lightweight local-only alternative when speed on CPU matters more
than multilingual support or dual-mode capability.

#### Snowflake Arctic-Embed (Large variant)

| Attribute | Value |
|-----------|-------|
| Parameters | 335M (also 23M small, 110M medium) |
| Dimensions | 1024 (Matryoshka support) |
| Context | Extended (variant-dependent) |
| Download | ~670 MB (large) / ~45 MB (small) |
| RAM/VRAM | ~2 GB (large) / <1 GB (small) |
| License | Apache 2.0 |
| Prefixes | Minimal query prefix |
| sentence-transformers | Yes |
| OpenRouter | **No** |

**Pros**: Three size variants (pick your tradeoff), Apache 2.0, Matryoshka
support, v2.0 adds multilingual support.
**Cons**: Not on OpenRouter (local only), not code-specialized, the small
variant sacrifices too much quality for code search.
**Best for**: When you want to precisely control the size/quality tradeoff.
Local only.

#### sentence-transformers/all-MiniLM-L6-v2

| Attribute | Value |
|-----------|-------|
| Parameters | 22M |
| Dimensions | 384 |
| Context | 512 tokens (OpenRouter) / 256 tokens (original) |
| Download | ~45 MB |
| RAM/VRAM | < 100 MB |
| License | Apache 2.0 |
| Prefixes | Not required |
| sentence-transformers | Yes (the original) |
| OpenRouter | Yes |
| CPU Speed | 5,000-14,000 sentences/sec |

**Pros**: Fastest model on the list, tiny footprint, no prefixes, universal
compatibility, massively battle-tested.
**Cons**: **256 token context** — truncates almost everything in a codebase.
384 dimensions limit retrieval quality. Low absolute accuracy. Essentially
unusable for code search where functions regularly exceed 256 tokens.
**Best for**: Toy projects, prototyping, or when you need sub-second indexing
of very small codebases.

---

### Tier 4: Code-Specialized (older, niche)

#### Salesforce CodeT5+ (110M embedding)

| Attribute | Value |
|-----------|-------|
| Parameters | 110M |
| Dimensions | 256 |
| Context | ~512 tokens |
| Download | ~220 MB |
| RAM/VRAM | ~500 MB |
| License | Apache 2.0 |
| Prefixes | [Match] token prefix |
| sentence-transformers | Via transformers (not native) |
| OpenRouter | No |

**Pros**: Purpose-built for code, lightweight, Apache 2.0.
**Cons**: Low 256 dimensions, 512 token context, dated architecture, not native
sentence-transformers support, lower benchmarks than modern models.
**Best for**: Legacy setups or when you need the smallest possible code-specific model.

#### Microsoft CodeBERT-base

| Attribute | Value |
|-----------|-------|
| Parameters | 110M |
| Dimensions | 768 |
| Context | 512 tokens |
| Download | ~440 MB |
| RAM/VRAM | ~1-2 GB |
| License | MIT |
| Prefixes | Not required |
| sentence-transformers | Via transformers |
| OpenRouter | No |
| CodeSearchNet MRR | 50.9% (vs 80%+ for modern models) |

**Pros**: MIT license, pioneered code understanding, still widely cited.
**Cons**: Significantly outperformed by modern models. 50.9% MRR vs 95%+ for
SFR-Code. Historical interest only.
**Best for**: Research, backward compatibility, or if MIT license is strictly required.

---

## Decision Matrix

| Priority | Best Choice | OpenRouter | Why |
|----------|------------|------------|-----|
| **Default (balanced)** | BGE-M3 | ✅ $0.01/M | Dual-mode, 8K context, multilingual, no prefix, Apache 2.0 |
| **Fastest on CPU** | nomic-embed-text-v1.5 | ❌ | 137M params, 8K context, Apache 2.0, local only |
| **Maximum code quality** | SFR-Embedding-Code-2B | ❌ | Top code benchmarks, 32K context, local only |
| **Maximum overall quality** | Qwen3-Embedding-8B | ✅ $0.01/M | #1 MTEB, strong code, dual-mode, needs GPU locally |
| **Code-specialized API** | Codestral Embed 2505 | ✅ $0.15/M | Built for code repos, 8K context, API-only |
| **Multilingual docs** | BGE-M3 | ✅ $0.01/M | 100+ languages, no prefixes, Apache 2.0 |
| **Offline / air-gapped** | nomic-embed-text-v1.5 | ❌ | 274MB download, runs on any CPU |
| **Minimal resources** | all-MiniLM-L6-v2 | ✅ $0.005/M | 45MB, but 512 token limit is severe for code |
| **Free API** | NVIDIA Nemotron Embed VL 1B | ✅ FREE | 131K context, multimodal, 1B params |
| **Longest context API** | Gemini Embedding 001 | ✅ $0.15/M | 20K context, top MTEB multilingual |
| **GPU with budget** | SFR-Embedding-Code-7B | ❌ | Absolute best code retrieval, local only |

---

## Multilingual + Polyglot Codebase Scenario

If your project contains documentation or code comments in multiple natural languages
(e.g., English, Russian, Ukrainian, Hungarian) and code spanning diverse programming
languages (e.g., Go, Rust, Java, C/C++, Ruby, PHP, bash), the default model choice
changes significantly. This section reassesses the candidates through that lens.

### Why This Scenario Is Hard

**Natural language diversity** — Russian and Ukrainian use Cyrillic script; Hungarian
is a Uralic language with complex morphology and limited representation in most
English-centric training corpora. A model trained primarily on English text will
produce low-quality embeddings for these languages, meaning semantic search across
language boundaries (e.g., searching in Russian for a concept documented in Ukrainian)
will fail or return poor results.

**Programming language breadth** — Most embedding benchmarks focus on Python and
JavaScript. Go, Rust, Ruby, PHP, and especially bash have distinct syntax patterns,
naming conventions, and idioms. A truly polyglot code model needs broad training
data across these ecosystems, not just the top-2 languages by GitHub popularity.

**Cross-lingual code-comment alignment** — The hardest case: a Go function with
Hungarian comments should be findable by a Russian-language query about the same
concept. This requires the model to align meaning across both natural and programming
language boundaries simultaneously.

### Reassessing the Top Candidates

| Model | EN Code | RU/UK Text | HU Text | Polyglot Code | OpenRouter | Context | CPU-viable | License |
|-------|---------|-----------|---------|---------------|------------|---------|------------|---------|
| **BGE-M3** | Good | **Strong** | **Good** | Good | ✅ $0.01/M | 8,192 | ⚠️ Slower | Apache 2.0 |
| **Qwen3-Embedding-0.6B** | Good | **Strong** | Likely decent | Good | ✅ Yes | 8,192 | ⚠️ Borderline | Proprietary |
| **Qwen3-Embedding-8B** | Excellent | **Excellent** | **Good** | Excellent | ✅ $0.01/M | 32,768 | ❌ GPU-only | Proprietary |
| **Gemini Embedding 001** | Good | **Strong** | **Good** | Good | ✅ $0.15/M | 20,480 | ❌ API-only | Proprietary |
| **multilingual-e5-large** | Good | **Strong** | **Good** | Moderate | ✅ $0.01/M | 512 | ✅ Yes | MIT |
| **nomic-embed-text-v1.5** | Good | Weak | Very Weak | Moderate | ❌ No | 8,192 | ✅ Fast | Apache 2.0 |
| **Jina v3** | Good | **Strong** | **Good** | Good | ❌ No | 8,192 | ⚠️ Slow | Proprietary |
| **nomic-embed-text-v2-moe** | Good | **Good** | **Decent** | Moderate | ❌ No | 512 | ✅ Yes | Apache 2.0 |
| **Llama-Embed-Nemotron-8B** | Excellent | **Excellent** | **Good** | Excellent | ❌ No | 32,768 | ❌ GPU-only | Apache 2.0 |
| **NVIDIA Nemotron Embed VL 1B** | Good | Moderate | Moderate | Good | ✅ FREE | 131,072 | ⚠️ Borderline | Open |

### Why nomic-embed-text-v1.5 Falls Short Here

Nomic v1.5 was trained primarily on English data. Its BERT-derived backbone gives it
some incidental multilingual capacity (BERT's original multilingual variant covers 104
languages), but the v1.5 fine-tuning optimized for English retrieval. In practice:

- **Russian**: Passable for common terms, weak for domain-specific vocabulary
- **Ukrainian**: Worse than Russian — less training data, Cyrillic overlap causes confusion
- **Hungarian**: Poor — agglutinative morphology and Latin-script Uralic vocabulary are
  far from the training distribution

When a developer searches in Russian for a function documented with Hungarian comments,
the semantic bridge simply isn't there.

### Revised Recommendation: BGE-M3

For multilingual + polyglot codebases, **BAAI/bge-m3** becomes the recommended default:

**Why BGE-M3 wins this scenario:**

1. **Explicit multilingual training on 100+ languages** — Russian, Ukrainian, and
   Hungarian are all represented in the training corpus. Cross-lingual retrieval
   (query in one language, match in another) is a design goal, not an accident.

2. **8,192 token context** — Same as Nomic v1.5, sufficient for large code functions.

3. **No prefix required** — Simpler integration. No risk of prefix misconfiguration
   breaking multilingual queries.

4. **Apache 2.0 license** — Same licensing freedom as Nomic v1.5.

5. **Hybrid retrieval modes** — Dense + sparse + multi-vector. The sparse mode
   (lexical matching) acts as a fallback when semantic matching is uncertain across
   languages — if the same variable name or function name appears in both query and
   document, sparse matching catches it even when dense embeddings are uncertain.

**The tradeoffs to accept:**

- **Larger model (568M vs 137M)** — ~2-3x slower on CPU. Indexing a 500-file project
  takes ~90-180 seconds instead of ~30-60. Still viable, just not instant.
- **Bigger download (~1.1 GB vs ~274 MB)** — First-time setup takes longer.
- **~1.5-2 GB RAM vs ~500 MB** — Still fits comfortably on any modern machine.

**Dual-mode confirmed:** BGE-M3 is available on OpenRouter at $0.01/M tokens with
8K context, AND locally via HuggingFace sentence-transformers. This makes it a true
dual-mode model — you can use OpenRouter for quick searches and HuggingFace for
offline/free indexing, or vice versa.

### Multilingual Configuration Example

```json
{
  "embedding": {
    "provider": "huggingface",
    "model": "BAAI/bge-m3",
    "dimensions": 1024,
    "query_prefix": "",
    "document_prefix": "",
    "batch_size": 32,
    "device": null
  }
}
```

### API Fallback: BGE-M3 on OpenRouter

Unlike the earlier analysis assumed, BGE-M3 is fully available on OpenRouter at
$0.01/M tokens with 8K context. This means you get true dual-mode out of the box:
use `"provider": "huggingface"` for free local indexing, and `"provider": "openrouter"`
for quick API-based searches on machines without Python or for CI/CD pipelines.
No need for a separate Jina provider — BGE-M3 covers both modes natively.

### When to Use Which Default

| Your Codebase | Recommended Model | Provider |
|---------------|-------------------|----------|
| English-only, speed priority | nomic-embed-text-v1.5 | huggingface (local only) |
| English-only, dual-mode needed | BAAI/bge-m3 | openrouter or huggingface |
| Multilingual (any combination) | BAAI/bge-m3 | openrouter or huggingface |
| Multilingual + maximum quality + GPU | Qwen3-Embedding-8B | openrouter or huggingface |
| Multilingual + free API | NVIDIA Nemotron Embed VL 1B v2 | openrouter (free) |
| Code-specialized + API budget | Codestral Embed 2505 | openrouter |

---

## How to Switch Models

In `.index/config.json`:

```json
{
  "embedding": {
    "provider": "huggingface",
    "model": "BAAI/bge-m3",
    "dimensions": 1024,
    "query_prefix": "",
    "document_prefix": "",
    "batch_size": 32
  }
}
```

Important: changing the model **invalidates the entire index**. The embedding
cache and vector store will be rebuilt from scratch on next `build_index.py` run,
because vectors from different models are not comparable.

Prefix configuration per model:

| Model | document_prefix | query_prefix |
|-------|----------------|--------------|
| nomic-embed-text-v1.5 | `"search_document: "` | `"search_query: "` |
| nomic-embed-text-v2-moe | `"search_document: "` | `"search_query: "` |
| BGE-M3 | `""` | `""` |
| BGE-Large-v1.5 | `""` | `""` |
| GTE-Large-v1.5 | `""` | `""` |
| Jina v3 | `""` | `""` (uses LoRA adapters internally) |
| all-MiniLM-L6-v2 | `""` | `""` |
| Snowflake Arctic | `""` | `"Represent this sentence for searching relevant passages: "` |
| SFR-Embedding-Code | `""` | Instruction-based (see model card) |
| Codestral Embed | `""` | `""` |
| Qwen3-Embedding | `""` | Instruction-based (see model card) |
| Gemini Embedding 001 | `""` | `""` |

---

## OpenRouter Available Embedding Models (Verified March 2026)

Complete list of embedding models available via OpenRouter's `/api/v1/embeddings` endpoint.
Use this table to pick a model for the `openrouter` provider.

| OpenRouter Model ID | Dimensions | Context | Price/M input | Multilingual | Code | Notes |
|---------------------|-----------|---------|---------------|-------------|------|-------|
| `nvidia/llama-nemotron-embed-vl-1b-v2` | — | 131K | **FREE** | Yes | Good | Multimodal (text+image), 1B params |
| `qwen/qwen3-embedding-8b` | configurable | 32K | $0.01 | **Yes** | Good | Top MTEB, needs GPU locally |
| `qwen/qwen3-embedding-4b` | configurable | 33K | $0.02 | **Yes** | Good | Mid-size Qwen variant |
| `qwen/qwen3-embedding-0.6b` | configurable | 8K | — | **Yes** | Good | Lightweight Qwen, dual-mode viable |
| `baai/bge-m3` | 1024 | 8K | $0.01 | **Yes (100+)** | Good | **DEFAULT** — dual-mode, Apache 2.0 |
| `baai/bge-large-en-v1.5` | 1024 | 512 | $0.01 | No | Moderate | English-only, short context |
| `baai/bge-base-en-v1.5` | 768 | 512 | $0.005 | No | Moderate | Smaller BGE variant |
| `mistralai/codestral-embed-2505` | — | 8K | $0.15 | No | **Yes** | Code-specialized |
| `mistralai/mistral-embed-2312` | 1024 | 8K | $0.10 | Moderate | No | General-purpose, older |
| `google/gemini-embedding-001` | — | 20K | $0.15 | **Yes** | Good | Top MTEB multilingual |
| `openai/text-embedding-3-large` | 3072 | 8K | $0.13 | Good | No | OpenAI's best embedding |
| `openai/text-embedding-3-small` | 1536 | 8K | $0.02 | Moderate | No | Cost-efficient OpenAI |
| `openai/text-embedding-ada-002` | 1536 | 8K | $0.10 | Moderate | No | Legacy, use 3-small instead |
| `intfloat/multilingual-e5-large` | 1024 | 512 | $0.01 | **Yes (90+)** | Moderate | Short context limits code use |
| `intfloat/e5-large-v2` | 1024 | 512 | $0.01 | No | Moderate | English-only E5 |
| `intfloat/e5-base-v2` | 768 | 512 | $0.005 | No | Moderate | Smaller E5 |
| `thenlper/gte-large` | 1024 | 512 | $0.01 | No | Moderate | English-only GTE |
| `thenlper/gte-base` | 768 | 512 | $0.005 | No | Moderate | Smaller GTE |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 512 | $0.005 | No | No | General-purpose |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | 512 | $0.005 | No | No | Lightweight |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 512 | $0.005 | No | No | Fastest, tiny |
| `sentence-transformers/paraphrase-MiniLM-L6-v2` | 384 | 512 | $0.005 | No | No | Paraphrase-optimized |
| `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 768 | 512 | $0.005 | No | No | QA-optimized |

**Key takeaways for this SKILL:**

- Only 6 models have context ≥ 8K (usable for code): BGE-M3, Qwen3 family, Mistral
  models, OpenAI models, Gemini, and NVIDIA Nemotron.
- Only 3 have both 8K+ context AND multilingual: BGE-M3 ($0.01), Qwen3 family ($0.01-0.02),
  and Gemini ($0.15).
- BGE-M3 is the cheapest multilingual option with 8K context on OpenRouter AND is
  available locally via HuggingFace — making it the strongest dual-mode default.
- The 512-context models (MiniLM, E5, GTE, BGE-en) are unsuitable for code search
  where functions regularly exceed 512 tokens.

**Models NOT on OpenRouter** (local-only via HuggingFace):
nomic-embed-text-v1.5, nomic-embed-text-v2-moe, SFR-Embedding-Code, Snowflake Arctic,
Jina v3, mxbai-embed-large-v1, CodeT5+, CodeBERT.

---

## Local Provider Setup (HuggingFace)

Phase 4 added the `huggingface` provider for local embedding inference via
sentence-transformers. This section covers setup, resource requirements, and
device selection.

### Installation

```bash
# Option 1: explicit flag
bash setup.sh --with-huggingface

# Option 2: auto-detected from config (if provider is "huggingface")
bash setup.sh
```

This installs `sentence-transformers>=3.0.0` and `torch>=2.0.0` into the
skill's virtual environment. First install is ~2-4 GB depending on platform.

### Model Download

On first use, the model is downloaded to `~/.cache/huggingface/hub`. Override
the cache location with the `HF_HUB_CACHE` environment variable.

| Model | Download Size | Disk Cache | RAM Usage |
|-------|--------------|------------|-----------|
| BAAI/bge-m3 (default) | ~1.1 GB | ~1.1 GB | ~1.5-2 GB |
| nomic-ai/nomic-embed-text-v1.5 | ~274 MB | ~274 MB | ~500 MB |
| SFR-Embedding-Code-400M | ~800 MB | ~800 MB | ~1.5 GB |

### Device Selection

The `embedding.device` config field controls where inference runs:

| Value | Behavior |
|-------|----------|
| `null` (default) | Auto-detect: CUDA > MPS > CPU |
| `"cpu"` | Force CPU (useful if GPU causes issues) |
| `"cuda"` | Force NVIDIA GPU (requires CUDA toolkit) |
| `"mps"` | Force Apple Silicon GPU (macOS only) |

### Performance Estimates

Approximate indexing speed for a 500-file project (~1,650 chunks):

| Device | BGE-M3 (568M) | Nomic v1.5 (137M) |
|--------|---------------|-------------------|
| CPU (modern laptop) | ~90-180 sec | ~30-60 sec |
| Apple M1/M2 (MPS) | ~20-40 sec | ~10-20 sec |
| NVIDIA GPU (CUDA) | ~5-15 sec | ~3-8 sec |

### Cross-Provider Compatibility

Indexes built with one provider can be searched with the other, as long as
the model and dimensions match. For example, you can build an index with
`"provider": "openrouter"` on a CI server and search locally with
`"provider": "huggingface"` — the vectors are identical.

### Reranking (Cross-Encoder)

Phase 4 also added optional cross-encoder reranking via the `search.rerank_*`
config fields. The default reranker model is `BAAI/bge-reranker-v2-m3` (~568 MB
download). Reranking requires the HuggingFace dependencies.

| Config Field | Default | Description |
|-------------|---------|-------------|
| `search.rerank_enabled` | `false` | Enable cross-encoder reranking |
| `search.rerank_model` | `"BAAI/bge-reranker-v2-m3"` | CrossEncoder model ID |
| `search.rerank_top_n` | `10` | Re-rank top N results |

Enable via config or CLI: `--rerank` flag on `semantic_search.py`.
