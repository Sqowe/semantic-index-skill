# Embedding Models for Semantic Index — Comprehensive Analysis

> This document compares embedding models suitable for code + documentation
> search, explains the default choice, and helps users pick the right model
> for their setup.

---

## Why nomic-ai/nomic-embed-text-v1.5 Is the Default

Before diving into the full comparison, here's the reasoning for the default choice.
The selection was driven by five hard constraints specific to this SKILL:

**1. Dual-mode compatibility (the killer constraint)**

The SKILL supports both OpenRouter (remote) and HuggingFace (local) providers.
The default model must work identically in both modes, producing the same vectors
so users can build an index with one provider and search with the other. Many
top models (SFR-Code-7B, Qwen3-Embedding-8B) are only available through one
channel. Nomic v1.5 runs on both OpenRouter and locally via sentence-transformers,
and produces bit-identical vectors either way.

**2. Context window ≥ 2048 tokens**

Code functions can be long. A 512-token context window (MiniLM, BGE-Large, E5,
mxbai) means chunks get truncated, losing the end of functions or the bottom
half of documentation sections. Nomic v1.5's 8,192-token window handles even
large class definitions without truncation. This is table-stakes for code search.

**3. CPU-viable performance**

Not everyone has a GPU. The model must index a 500-file project in under
60 seconds on a modern laptop CPU. At 137M parameters and ~500-750 sentences/sec
on CPU, Nomic v1.5 hits this target. Models over 1B parameters (BGE-M3 at 335M
is borderline; Jina v3 at 570M is slow; anything 7B+ is GPU-only).

**4. Apache 2.0 license**

The SKILL is designed to be freely distributable. The default model must be
usable without licensing concerns. This eliminates Jina v3 (proprietary),
mxbai-embed (proprietary), and some Qwen variants.

**5. Proven quality at this parameter class**

At 137M parameters, Nomic v1.5 consistently outperforms or matches models
2-3x its size on MTEB retrieval benchmarks. Its Matryoshka support means
users can reduce dimensions from 768 to 256 or even 64 with graceful
quality degradation — useful for very large projects where index size matters.

### What Nomic v1.5 Is NOT Best At

To be transparent about the tradeoffs:

- **Not the highest absolute quality**: 7-8B models (SFR-Code-7B, Qwen3-Embedding-8B)
  score 10-15% higher on code retrieval benchmarks. If you have a GPU and only
  use local mode, those are better.
- **Not code-specialized**: It's a general-purpose text embedding model. Models
  like SFR-Embedding-Code and CodeT5+ are trained specifically on code corpora
  and understand programming constructs better.
- **Requires task prefixes**: You must prepend `search_document:` to documents
  and `search_query:` to queries. Other models (BGE, E5, mxbai) work without
  any prefix, which is simpler. The SKILL handles this automatically, but it's
  a gotcha if someone uses the index outside the SKILL.
- **Not multilingual**: If your codebase has documentation in multiple natural
  languages, BGE-M3 or Nomic v2-MoE are better choices.
- **v2-MoE exists and is newer**: Released February 2025, Nomic v2-MoE uses
  Mixture-of-Experts (475M total / 305M active) with multilingual support
  and improved benchmarks. However, it has a shorter context window (512 tokens
  — a dealbreaker for code) and less production track record. Once v2 gets an
  8K context variant, it would likely replace v1.5 as the default.

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
| OpenRouter | Yes |
| MTEB Rank | #1 Multilingual MTEB (Oct 2025) |

**Pros**: Highest multilingual benchmark scores, massive context window, excellent
code understanding from Llama 3.1 base.
**Cons**: 16 GB VRAM means high-end GPU only. Overkill for most project-level
code search. Slow on CPU (unusable).
**Best for**: Large enterprise codebases with multilingual documentation, when
GPU infrastructure is available.

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
| OpenRouter | Yes |
| MTEB Rank | #1 Multilingual MTEB (June 2025, score 70.58) |

**Pros**: Top MTEB scores, strong code retrieval, configurable output dimensions,
Flash Attention 2 support.
**Cons**: Proprietary license, massive memory footprint, GPU-only.
**Best for**: Teams with GPU clusters who need maximum retrieval quality.

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

#### BAAI BGE-M3

| Attribute | Value |
|-----------|-------|
| Parameters | 335M |
| Dimensions | 1024 |
| Context | 8,192 tokens |
| Download | ~670 MB |
| RAM/VRAM | ~1-2 GB |
| License | Apache 2.0 |
| Prefixes | Not required |
| sentence-transformers | Yes |
| OpenRouter | Yes |
| MTEB Score | ~63.0 average |

**Pros**: 100+ language support, hybrid retrieval (dense + sparse + multi-vector),
no prefix needed (simpler integration), good context window, Apache 2.0.
**Cons**: Larger than Nomic v1.5 (335M vs 137M) so 2-3x slower on CPU. Not
code-specialized. Multi-vector mode adds complexity.
**Best for**: Multilingual codebases, projects with docs in multiple languages.

#### Alibaba GTE-Large-En-v1.5

| Attribute | Value |
|-----------|-------|
| Parameters | 434M |
| Dimensions | 1024 |
| Context | 8,192 tokens |
| Download | ~850 MB |
| RAM/VRAM | ~2 GB |
| License | Open source |
| Prefixes | Not required |
| sentence-transformers | Yes |
| OpenRouter | Limited |
| MTEB Score | 65.39 |

**Pros**: Strong retrieval scores, 8K context, no prefixes needed, clean
integration.
**Cons**: Larger model, slower on CPU. English-only. Limited OpenRouter availability
breaks dual-mode.
**Best for**: English-only projects where you prioritize simplicity (no prefixes)
and have moderate compute.

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
| OpenRouter | Yes |
| MTEB Score | 64.44 (multilingual) |

**Pros**: Task-specific LoRA adapters (different embedding strategies for
retrieval vs. clustering vs. classification), MRL dimension reduction, good
multilingual support.
**Cons**: Proprietary license, 570M is heavy for CPU, LoRA adapter switching
adds complexity, not code-specialized.
**Best for**: Multi-purpose embedding needs (not just search but also
clustering/classification).

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
| OpenRouter | Limited |
| MTEB Score | Competitive with BGE variants |

**Pros**: No prefix needed, binary quantization support (extreme compression),
MRL dimension reduction.
**Cons**: 512 token context (too short for code), proprietary license, limited
OpenRouter support.
**Best for**: Text-heavy projects where storage efficiency matters more than
code search quality.

---

### Tier 3: Lightweight (< 150M params, fast CPU inference)

Best for resource-constrained environments or when speed matters most.

#### nomic-ai/nomic-embed-text-v1.5 (DEFAULT)

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
| OpenRouter | Yes |
| Downloads | 9.7M (most downloaded in family) |
| CPU Speed | ~500-750 sentences/sec |

**Pros**: Best quality-to-size ratio in its class, 8K context window (critical
for code), Matryoshka dimensions, Apache 2.0, dual-mode compatible (OpenRouter +
local), very fast on CPU, tiny download, proven in production (9.7M downloads).
**Cons**: Requires task prefixes, not code-specialized, English-only, lower
absolute quality than 335M+ models.
**Best for**: The default choice for most projects. Balanced across all criteria.

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
| OpenRouter | Limited |

**Pros**: Three size variants (pick your tradeoff), Apache 2.0, Matryoshka
support, v2.0 adds multilingual support.
**Cons**: Limited OpenRouter availability, not code-specialized, the small
variant sacrifices too much quality for code search.
**Best for**: When you want to precisely control the size/quality tradeoff.

#### sentence-transformers/all-MiniLM-L6-v2

| Attribute | Value |
|-----------|-------|
| Parameters | 22M |
| Dimensions | 384 |
| Context | 256 tokens |
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

| Priority | Best Choice | Why |
|----------|------------|-----|
| **Default (balanced)** | nomic-embed-text-v1.5 | Dual-mode, 8K context, fast CPU, Apache 2.0 |
| **Maximum code quality** | SFR-Embedding-Code-2B | Top code benchmarks, 32K context |
| **Maximum overall quality** | Qwen3-Embedding-8B | #1 MTEB, strong code, needs GPU |
| **Multilingual docs** | BGE-M3 | 100+ languages, no prefixes, Apache 2.0 |
| **Offline / air-gapped** | nomic-embed-text-v1.5 | 274MB download, runs on any CPU |
| **Minimal resources** | all-MiniLM-L6-v2 | 45MB, but 256 token limit is severe |
| **GPU with budget** | SFR-Embedding-Code-7B | Absolute best code retrieval |
| **Multilingual + lightweight** | nomic-embed-text-v2-moe | 305M active, 100+ langs, but 512 context |

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

| Model | EN Code | RU/UK Text | HU Text | Polyglot Code | Dual-mode | Context | CPU-viable | License |
|-------|---------|-----------|---------|---------------|-----------|---------|------------|---------|
| **nomic-embed-text-v1.5** | Good | Weak | Very Weak | Moderate | ✅ Yes | 8,192 | ✅ Fast | Apache 2.0 |
| **BGE-M3** | Good | **Strong** | **Good** | Good | ⚠️ Limited | 8,192 | ⚠️ Slower | Apache 2.0 |
| **Jina v3** | Good | **Strong** | **Good** | Good | ❌ Proprietary | 8,192 | ⚠️ Slow | Proprietary |
| **multilingual-e5-large-instruct** | Good | **Strong** | **Good** | Moderate | ❌ No OR | 514 | ✅ Yes | MIT |
| **Qwen3-Embedding-0.6B** | Good | **Strong** | Likely decent | Good | ❌ No OR | 32,768 | ⚠️ Borderline | Proprietary |
| **nomic-embed-text-v2-moe** | Good | **Good** | **Decent** | Moderate | ❌ No OR | 512 | ✅ Yes | Apache 2.0 |
| **Llama-Embed-Nemotron-8B** | Excellent | **Excellent** | **Good** | Excellent | ✅ Yes | 32,768 | ❌ GPU-only | Apache 2.0 |

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
- **Limited OpenRouter availability** — BGE-M3 is listed on OpenRouter but with
  inconsistent availability. For this scenario, **local-first with HuggingFace
  provider is the right default** — you avoid API costs entirely and get the best
  multilingual quality.
- **No dual-mode guarantee** — If you need API+local with identical vectors, BGE-M3
  is riskier than Nomic v1.5. But for multilingual use, the quality gain far
  outweighs the dual-mode convenience.

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

### Alternative: Jina v3 (If You Want an API Fallback)

If having an API provider matters (e.g., for CI/CD pipelines or machines without
Python), **Jina v3** (`jinaai/jina-embeddings-v3`) offers excellent multilingual
quality with its own API (api.jina.ai). This would require adding a `jina` provider
to the SKILL alongside `openrouter` and `huggingface`. The trade-off: proprietary
license and 570M parameters (slower on CPU than BGE-M3's 568M due to architecture
differences).

### When to Use Which Default

| Your Codebase | Recommended Model | Provider |
|---------------|-------------------|----------|
| English-only code + English docs | nomic-embed-text-v1.5 | openrouter or huggingface |
| English code + 2+ natural languages in docs/comments | BAAI/bge-m3 | huggingface |
| Multilingual everything + GPU available | Llama-Embed-Nemotron-8B | huggingface |
| Multilingual + need API fallback | jinaai/jina-embeddings-v3 | huggingface + jina API |

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
