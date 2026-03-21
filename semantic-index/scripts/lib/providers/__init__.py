"""Embedding provider implementations.

Each provider module implements the EmbeddingProvider ABC defined
in lib.embedder. Providers are loaded lazily by the create_embedder()
factory — only the selected provider's dependencies are imported.

Available providers:
    - openrouter: REST API via OpenRouter (requires API key)
    - huggingface: Local inference via sentence-transformers (no API key)
"""

PROVIDER_REGISTRY: dict[str, str] = {
    "openrouter": "lib.providers.openrouter.OpenRouterProvider",
    "huggingface": "lib.providers.huggingface.HuggingFaceProvider",
}
