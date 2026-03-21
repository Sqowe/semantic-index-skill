"""Tests for the DITA XML chunker (Phase 5).

Covers: basic topic chunking, nested topics, specializations via class
attribute, namespaced XML, ditamap parsing, prolog metadata extraction,
conref detection, xml:lang propagation, oversized section splitting,
parse failure handling, and config migration edge cases.
"""

import pytest

from lib.chunkers.dita import chunk_dita
from lib.models import ChunkType


# ---------------------------------------------------------------------------
# XML fixtures
# ---------------------------------------------------------------------------

CONCEPT_TOPIC = """\
<?xml version="1.0" encoding="UTF-8"?>
<concept id="about">
  <title>About the Product</title>
  <shortdesc>A brief overview of the product.</shortdesc>
  <prolog>
    <metadata>
      <keywords><keyword>overview</keyword><keyword>product</keyword></keywords>
      <audience type="admin"/>
      <category>Getting Started</category>
    </metadata>
    <author>Jane Doe</author>
  </prolog>
  <conbody>
    <p>This product helps you manage your infrastructure.</p>
    <p>It supports multiple cloud providers and on-premise deployments.</p>
  </conbody>
</concept>
"""

TASK_TOPIC = """\
<?xml version="1.0" encoding="UTF-8"?>
<task id="install">
  <title>Install the Software</title>
  <shortdesc>Steps to install the software on your machine.</shortdesc>
  <taskbody>
    <prereq>You need admin access to the target machine.</prereq>
    <steps>
      <step><cmd>Download the installer from the website.</cmd></step>
      <step><cmd>Run the installer with elevated privileges.</cmd></step>
      <step><cmd>Follow the on-screen instructions.</cmd></step>
    </steps>
    <result>The software is now installed and ready to use.</result>
  </taskbody>
</task>
"""

NESTED_TOPICS = """\
<?xml version="1.0" encoding="UTF-8"?>
<topic id="parent">
  <title>Parent Topic</title>
  <body>
    <p>This is the parent topic content with enough words to pass the minimum token threshold easily.</p>
  </body>
  <topic id="child1">
    <title>Child Topic One</title>
    <body>
      <p>First child topic content with enough words to pass the minimum token threshold easily here.</p>
    </body>
  </topic>
  <topic id="child2">
    <title>Child Topic Two</title>
    <body>
      <p>Second child topic content with enough words to pass the minimum token threshold easily here.</p>
    </body>
  </topic>
</topic>
"""

SPECIALIZED_TOPIC = """\
<?xml version="1.0" encoding="UTF-8"?>
<myCustomTopic id="spec1" class="- topic/topic myDomain/myCustomTopic ">
  <myTitle class="- topic/title myDomain/myTitle ">Specialized Title</myTitle>
  <myShortdesc class="- topic/shortdesc myDomain/myShortdesc ">Specialized short description here.</myShortdesc>
  <myBody class="- topic/body myDomain/myBody ">
    <p>Specialized body content with enough words to pass the minimum token threshold for chunking.</p>
  </myBody>
</myCustomTopic>
"""

NAMESPACED_TOPIC = """\
<?xml version="1.0" encoding="UTF-8"?>
<ns:concept xmlns:ns="http://example.com/dita" id="ns-topic" class="- topic/topic concept/concept ">
  <ns:title class="- topic/title ">Namespaced Title</ns:title>
  <ns:shortdesc class="- topic/shortdesc ">Namespaced short description for testing.</ns:shortdesc>
  <ns:conbody class="- topic/body concept/conbody ">
    <ns:p>Namespaced body content with enough words to pass the minimum token threshold for chunking.</ns:p>
  </ns:conbody>
</ns:concept>
"""

TOPIC_WITH_CONREF = """\
<?xml version="1.0" encoding="UTF-8"?>
<concept id="reuse">
  <title>Reusable Content Topic</title>
  <conbody>
    <p conref="shared/common.dita#common/intro">Placeholder for reused content that will be pulled from another file at build time.</p>
    <p>Local content that is not reused from another file and provides additional context for the reader.</p>
    <p>This paragraph adds enough words to ensure the topic exceeds the minimum token threshold for chunking.</p>
  </conbody>
</concept>
"""

TOPIC_WITH_LANG = """\
<?xml version="1.0" encoding="UTF-8"?>
<topic id="multilang" xml:lang="de-DE">
  <title>German Topic Title</title>
  <body>
    <p>Dies ist ein deutscher Absatz mit genug Wörtern um den Mindest-Token-Schwellenwert zu überschreiten.</p>
  </body>
</topic>
"""

TOPIC_WITH_SECTIONS = """\
<?xml version="1.0" encoding="UTF-8"?>
<concept id="big">
  <title>Big Topic</title>
  <shortdesc>Overview of a large topic.</shortdesc>
  <conbody>
    <p>Introduction paragraph before any sections with enough content to be meaningful.</p>
    <section>
      <title>Section One</title>
      <p>Content of section one with enough words to pass the minimum token threshold for chunking.</p>
    </section>
    <section>
      <title>Section Two</title>
      <p>Content of section two with enough words to pass the minimum token threshold for chunking.</p>
    </section>
  </conbody>
</concept>
"""

DITAMAP_BASIC = """\
<?xml version="1.0" encoding="UTF-8"?>
<map>
  <title>Product Documentation</title>
  <topicref href="intro.dita" navtitle="Introduction">
    <topicref href="install.dita" navtitle="Installation"/>
    <topicref href="config.dita" navtitle="Configuration"/>
  </topicref>
  <topicref href="reference.dita" keys="ref-guide">
    <topicmeta><navtitle>Reference Guide</navtitle></topicmeta>
  </topicref>
</map>
"""

DITAMAP_WITH_LANG = """\
<?xml version="1.0" encoding="UTF-8"?>
<map xml:lang="ja-JP">
  <title>Japanese Documentation Map</title>
  <topicref href="overview.dita" navtitle="概要"/>
  <topicref href="setup.dita" navtitle="セットアップ"/>
</map>
"""

INVALID_XML = """<broken><unclosed>"""

EMPTY_TOPIC = """\
<?xml version="1.0" encoding="UTF-8"?>
<topic id="empty">
  <title></title>
  <body></body>
</topic>
"""


# ---------------------------------------------------------------------------
# Basic topic chunking
# ---------------------------------------------------------------------------

class TestBasicTopicChunking:
    """Test chunking of standard DITA topic types."""

    def test_concept_produces_chunk(self, default_config):
        chunks = chunk_dita(CONCEPT_TOPIC, "docs/about.dita", "dita", default_config)
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == ChunkType.DITA_TOPIC
        assert chunks[0].language == "dita"
        assert "About the Product" in chunks[0].content

    def test_task_produces_chunk(self, default_config):
        chunks = chunk_dita(TASK_TOPIC, "docs/install.dita", "dita", default_config)
        assert len(chunks) >= 1
        assert "Install the Software" in chunks[0].content
        assert "Download the installer" in chunks[0].content

    def test_concept_has_symbol_name(self, default_config):
        chunks = chunk_dita(CONCEPT_TOPIC, "docs/about.dita", "dita", default_config)
        assert chunks[0].symbol_name == "About the Product"

    def test_concept_has_file_path(self, default_config):
        chunks = chunk_dita(CONCEPT_TOPIC, "docs/about.dita", "dita", default_config)
        assert chunks[0].file_path == "docs/about.dita"

    def test_empty_topic_produces_no_chunks(self, default_config):
        chunks = chunk_dita(EMPTY_TOPIC, "docs/empty.dita", "dita", default_config)
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Prolog metadata extraction
# ---------------------------------------------------------------------------

class TestPrologMetadata:
    """Test extraction of prolog metadata into chunk metadata."""

    def test_keywords_in_metadata(self, default_config):
        chunks = chunk_dita(CONCEPT_TOPIC, "docs/about.dita", "dita", default_config)
        meta = chunks[0].metadata
        assert "prolog" in meta
        assert meta["prolog"].get("keywords") == ["overview", "product"]

    def test_audience_in_metadata(self, default_config):
        chunks = chunk_dita(CONCEPT_TOPIC, "docs/about.dita", "dita", default_config)
        assert chunks[0].metadata["prolog"].get("audience") == "admin"

    def test_category_in_metadata(self, default_config):
        chunks = chunk_dita(CONCEPT_TOPIC, "docs/about.dita", "dita", default_config)
        assert chunks[0].metadata["prolog"].get("category") == "Getting Started"

    def test_prolog_prefix_in_content(self, default_config):
        chunks = chunk_dita(CONCEPT_TOPIC, "docs/about.dita", "dita", default_config)
        content = chunks[0].content
        assert "[audience: admin]" in content
        assert "[keywords: overview, product]" in content

    def test_topic_type_in_metadata(self, default_config):
        chunks = chunk_dita(CONCEPT_TOPIC, "docs/about.dita", "dita", default_config)
        assert chunks[0].metadata["topic_type"] == "concept"


# ---------------------------------------------------------------------------
# Nested topics
# ---------------------------------------------------------------------------

class TestNestedTopics:
    """Test that nested child topics are chunked alongside the parent."""

    def test_parent_and_children_all_chunked(self, default_config):
        chunks = chunk_dita(NESTED_TOPICS, "docs/nested.dita", "dita", default_config)
        titles = [c.symbol_name for c in chunks]
        assert "Parent Topic" in titles
        assert "Child Topic One" in titles
        assert "Child Topic Two" in titles

    def test_nested_produces_three_chunks(self, default_config):
        chunks = chunk_dita(NESTED_TOPICS, "docs/nested.dita", "dita", default_config)
        assert len(chunks) == 3

    def test_parent_body_excludes_child_content(self, default_config):
        chunks = chunk_dita(NESTED_TOPICS, "docs/nested.dita", "dita", default_config)
        parent = next(c for c in chunks if c.symbol_name == "Parent Topic")
        assert "First child" not in parent.content
        assert "Second child" not in parent.content


# ---------------------------------------------------------------------------
# Specializations via class attribute
# ---------------------------------------------------------------------------

class TestSpecializations:
    """Test that specialized topic types are detected via class attribute."""

    def test_specialized_topic_detected(self, default_config):
        chunks = chunk_dita(SPECIALIZED_TOPIC, "docs/spec.dita", "dita", default_config)
        assert len(chunks) >= 1

    def test_specialized_title_extracted(self, default_config):
        chunks = chunk_dita(SPECIALIZED_TOPIC, "docs/spec.dita", "dita", default_config)
        assert chunks[0].symbol_name == "Specialized Title"

    def test_specialized_body_extracted(self, default_config):
        chunks = chunk_dita(SPECIALIZED_TOPIC, "docs/spec.dita", "dita", default_config)
        assert "Specialized body content" in chunks[0].content

    def test_specialized_shortdesc_extracted(self, default_config):
        chunks = chunk_dita(SPECIALIZED_TOPIC, "docs/spec.dita", "dita", default_config)
        assert "Specialized short description" in chunks[0].content


# ---------------------------------------------------------------------------
# Namespaced XML
# ---------------------------------------------------------------------------

class TestNamespacedXML:
    """Test that namespaced DITA elements are handled correctly."""

    def test_namespaced_topic_detected(self, default_config):
        chunks = chunk_dita(NAMESPACED_TOPIC, "docs/ns.dita", "dita", default_config)
        assert len(chunks) >= 1

    def test_namespaced_title_extracted(self, default_config):
        chunks = chunk_dita(NAMESPACED_TOPIC, "docs/ns.dita", "dita", default_config)
        assert chunks[0].symbol_name == "Namespaced Title"

    def test_namespaced_body_extracted(self, default_config):
        chunks = chunk_dita(NAMESPACED_TOPIC, "docs/ns.dita", "dita", default_config)
        assert "Namespaced body content" in chunks[0].content


# ---------------------------------------------------------------------------
# Content references (conref)
# ---------------------------------------------------------------------------

class TestConrefs:
    """Test conref/conkeyref detection in metadata."""

    def test_conref_noted_in_metadata(self, default_config):
        chunks = chunk_dita(TOPIC_WITH_CONREF, "docs/reuse.dita", "dita", default_config)
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("has_conrefs") is True

    def test_no_conref_when_absent(self, default_config):
        chunks = chunk_dita(CONCEPT_TOPIC, "docs/about.dita", "dita", default_config)
        assert "has_conrefs" not in chunks[0].metadata


# ---------------------------------------------------------------------------
# xml:lang propagation
# ---------------------------------------------------------------------------

class TestXmlLang:
    """Test xml:lang attribute extraction into metadata."""

    def test_lang_in_metadata(self, default_config):
        chunks = chunk_dita(TOPIC_WITH_LANG, "docs/de.dita", "dita", default_config)
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("xml_lang") == "de-DE"

    def test_ditamap_lang_in_metadata(self, default_config):
        chunks = chunk_dita(DITAMAP_WITH_LANG, "docs/ja.ditamap", "ditamap", default_config)
        if chunks:  # may be empty if below min_tokens
            assert chunks[0].metadata.get("xml_lang") == "ja-JP"


# ---------------------------------------------------------------------------
# Oversized topic splitting
# ---------------------------------------------------------------------------

class TestOversizedSplitting:
    """Test section-level splitting when a topic exceeds max_tokens."""

    def test_splits_at_sections(self, small_config):
        chunks = chunk_dita(TOPIC_WITH_SECTIONS, "docs/big.dita", "dita", small_config)
        # Should produce multiple chunks (intro + sections)
        assert len(chunks) >= 2

    def test_section_chunks_have_title_context(self, small_config):
        chunks = chunk_dita(TOPIC_WITH_SECTIONS, "docs/big.dita", "dita", small_config)
        for chunk in chunks:
            assert "Big Topic" in chunk.content

    def test_section_index_in_metadata(self, small_config):
        chunks = chunk_dita(TOPIC_WITH_SECTIONS, "docs/big.dita", "dita", small_config)
        section_chunks = [c for c in chunks if "section_index" in c.metadata]
        assert len(section_chunks) >= 1
        indices = [c.metadata["section_index"] for c in section_chunks]
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# Ditamap parsing
# ---------------------------------------------------------------------------

class TestDitamapChunking:
    """Test .ditamap file chunking."""

    def test_ditamap_produces_map_chunk(self, default_config):
        chunks = chunk_dita(DITAMAP_BASIC, "docs/product.ditamap", "ditamap", default_config)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.DITA_MAP

    def test_ditamap_has_title(self, default_config):
        chunks = chunk_dita(DITAMAP_BASIC, "docs/product.ditamap", "ditamap", default_config)
        assert chunks[0].symbol_name == "Product Documentation"
        assert "Map: Product Documentation" in chunks[0].content

    def test_ditamap_has_navtitles(self, default_config):
        chunks = chunk_dita(DITAMAP_BASIC, "docs/product.ditamap", "ditamap", default_config)
        content = chunks[0].content
        assert "Introduction" in content
        assert "Installation" in content
        assert "Configuration" in content

    def test_ditamap_topicmeta_navtitle(self, default_config):
        chunks = chunk_dita(DITAMAP_BASIC, "docs/product.ditamap", "ditamap", default_config)
        assert "Reference Guide" in chunks[0].content

    def test_ditamap_has_hrefs(self, default_config):
        chunks = chunk_dita(DITAMAP_BASIC, "docs/product.ditamap", "ditamap", default_config)
        assert "(intro.dita)" in chunks[0].content
        assert "(install.dita)" in chunks[0].content

    def test_ditamap_has_keys(self, default_config):
        chunks = chunk_dita(DITAMAP_BASIC, "docs/product.ditamap", "ditamap", default_config)
        assert "[keys: ref-guide]" in chunks[0].content

    def test_ditamap_language_is_ditamap(self, default_config):
        chunks = chunk_dita(DITAMAP_BASIC, "docs/product.ditamap", "ditamap", default_config)
        assert chunks[0].language == "ditamap"

    def test_ditamap_nested_indentation(self, default_config):
        chunks = chunk_dita(DITAMAP_BASIC, "docs/product.ditamap", "ditamap", default_config)
        content = chunks[0].content
        # Child topicrefs should be indented
        assert "  - Installation" in content


# ---------------------------------------------------------------------------
# Parse failure handling
# ---------------------------------------------------------------------------

class TestParseFailure:
    """Test graceful handling of invalid XML."""

    def test_invalid_xml_returns_empty(self, default_config):
        chunks = chunk_dita(INVALID_XML, "docs/broken.dita", "dita", default_config)
        assert chunks == []

    def test_empty_string_returns_empty(self, default_config):
        chunks = chunk_dita("", "docs/empty.dita", "dita", default_config)
        assert chunks == []


# ---------------------------------------------------------------------------
# Language detection integration
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    """Test that .dita and .ditamap are detected by the language mapper."""

    def test_dita_extension(self):
        from lib.chunkers.common import detect_language
        assert detect_language("docs/topic.dita") == "dita"

    def test_ditamap_extension(self):
        from lib.chunkers.common import detect_language
        assert detect_language("docs/product.ditamap") == "ditamap"


# ---------------------------------------------------------------------------
# Config migration edge case
# ---------------------------------------------------------------------------

class TestMigrationDitaExtensions:
    """Test migrate_config handles DITA extension addition correctly."""

    def test_adds_dita_to_existing_extensions(self):
        from scripts.migrate_config import analyze_config
        config = {
            "schema_version": "1.0",
            "indexing": {"file_extensions": [".py", ".md"]},
            "search": dict(
                default_top_k=10, default_threshold=0.3, mode="hybrid",
                hybrid_alpha=0.7, rerank_enabled=False,
                rerank_model="BAAI/bge-reranker-v2-m3", rerank_top_n=10,
            ),
            "embedding": {"device": None, "trust_remote_code": False},
        }
        migrations = analyze_config(config)
        dita_mig = [m for m in migrations if "DITA" in m.get("reason", "")]
        assert len(dita_mig) == 1
        assert ".dita" in dita_mig[0]["new_value"]
        assert ".ditamap" in dita_mig[0]["new_value"]

    def test_adds_dita_to_empty_extensions(self):
        from scripts.migrate_config import analyze_config
        config = {
            "schema_version": "1.0",
            "indexing": {"file_extensions": []},
            "search": dict(
                default_top_k=10, default_threshold=0.3, mode="hybrid",
                hybrid_alpha=0.7, rerank_enabled=False,
                rerank_model="BAAI/bge-reranker-v2-m3", rerank_top_n=10,
            ),
            "embedding": {"device": None, "trust_remote_code": False},
        }
        migrations = analyze_config(config)
        dita_mig = [m for m in migrations if "DITA" in m.get("reason", "")]
        assert len(dita_mig) == 1
        assert dita_mig[0]["new_value"] == [".dita", ".ditamap"]

    def test_skips_when_dita_already_present(self):
        from scripts.migrate_config import analyze_config
        config = {
            "schema_version": "1.0",
            "indexing": {"file_extensions": [".py", ".dita", ".ditamap"]},
            "search": dict(
                default_top_k=10, default_threshold=0.3, mode="hybrid",
                hybrid_alpha=0.7, rerank_enabled=False,
                rerank_model="BAAI/bge-reranker-v2-m3", rerank_top_n=10,
            ),
            "embedding": {"device": None, "trust_remote_code": False},
        }
        migrations = analyze_config(config)
        dita_mig = [m for m in migrations if "DITA" in m.get("reason", "")]
        assert len(dita_mig) == 0


# ---------------------------------------------------------------------------
# Oversized section fixtures
# ---------------------------------------------------------------------------

def _make_huge_section(word_count: int) -> str:
    """Generate a DITA section with approximately word_count words."""
    words = " ".join(f"word{i}" for i in range(word_count))
    return f"<section><title>Huge Section</title><p>{words}</p></section>"


def _make_large_ditamap(ref_count: int) -> str:
    """Generate a ditamap with ref_count topicrefs."""
    refs = "\n".join(
        f'  <topicref href="topic{i}.dita" navtitle="Topic Number {i} Title"/>'
        for i in range(ref_count)
    )
    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<map>
  <title>Large Map</title>
{refs}
</map>
"""


TOPIC_WITH_HUGE_SECTION = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<concept id="huge">
  <title>Huge Section Topic</title>
  <conbody>
{_make_huge_section(800)}
  </conbody>
</concept>
"""


# ---------------------------------------------------------------------------
# max_tokens enforcement
# ---------------------------------------------------------------------------

class TestMaxTokensEnforcement:
    """Test that chunks never exceed max_tokens."""

    def test_oversized_section_capped(self, small_config):
        """A section with 800 words should be truncated to fit max_tokens."""
        chunks = chunk_dita(
            TOPIC_WITH_HUGE_SECTION, "docs/huge.dita", "dita", small_config,
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.token_count <= small_config.chunking.max_tokens

    def test_oversized_section_has_truncated_flag(self, small_config):
        chunks = chunk_dita(
            TOPIC_WITH_HUGE_SECTION, "docs/huge.dita", "dita", small_config,
        )
        truncated = [c for c in chunks if c.metadata.get("truncated")]
        assert len(truncated) >= 1

    def test_large_ditamap_capped(self, small_config):
        """A ditamap with 200 topicrefs should be truncated."""
        xml = _make_large_ditamap(200)
        chunks = chunk_dita(xml, "docs/big.ditamap", "ditamap", small_config)
        assert len(chunks) == 1
        assert chunks[0].token_count <= small_config.chunking.max_tokens

    def test_large_ditamap_has_truncated_flag(self, small_config):
        xml = _make_large_ditamap(200)
        chunks = chunk_dita(xml, "docs/big.ditamap", "ditamap", small_config)
        assert chunks[0].metadata.get("truncated") is True


# ---------------------------------------------------------------------------
# XML safety guard
# ---------------------------------------------------------------------------

class TestXMLSafetyGuard:
    """Test that oversized XML content is rejected gracefully."""

    def test_oversized_xml_returns_empty(self, default_config):
        """Content exceeding 10 MB safety limit should return []."""
        huge_xml = '<?xml version="1.0"?><topic id="x"><title>X</title><body>'
        huge_xml += "<p>" + ("A " * 6_000_000) + "</p>"
        huge_xml += "</body></topic>"
        chunks = chunk_dita(huge_xml, "docs/bomb.dita", "dita", default_config)
        assert chunks == []


# ---------------------------------------------------------------------------
# Line approximate metadata
# ---------------------------------------------------------------------------

class TestLineApproximateMetadata:
    """Test that chunks include line_approximate flag."""

    def test_topic_has_line_approximate(self, default_config):
        chunks = chunk_dita(CONCEPT_TOPIC, "docs/about.dita", "dita", default_config)
        assert chunks[0].metadata.get("line_approximate") is True

    def test_ditamap_has_line_approximate(self, default_config):
        chunks = chunk_dita(DITAMAP_BASIC, "docs/product.ditamap", "ditamap", default_config)
        assert chunks[0].metadata.get("line_approximate") is True


# ---------------------------------------------------------------------------
# Migration type robustness
# ---------------------------------------------------------------------------

class TestMigrationTypeRobustness:
    """Test that migration handles non-list file_extensions gracefully."""

    def test_string_extensions_normalized(self):
        from scripts.migrate_config import analyze_config
        config = {
            "schema_version": "1.0",
            "indexing": {"file_extensions": ".py"},
            "search": dict(
                default_top_k=10, default_threshold=0.3, mode="hybrid",
                hybrid_alpha=0.7, rerank_enabled=False,
                rerank_model="BAAI/bge-reranker-v2-m3", rerank_top_n=10,
            ),
            "embedding": {"device": None, "trust_remote_code": False},
        }
        migrations = analyze_config(config)
        dita_mig = [m for m in migrations if "DITA" in m.get("reason", "")]
        assert len(dita_mig) == 1
        assert isinstance(dita_mig[0]["new_value"], list)


    def test_single_line_oversize_truncated(self):
        """A single very long line with no newlines must still be capped."""
        from lib.chunkers.dita import chunk_dita
        from lib.config import Config, ChunkingConfig
        cfg = Config()
        cfg.chunking = ChunkingConfig(max_tokens=30, overlap_tokens=5, min_tokens=5)
        words = " ".join(f"word{i}" for i in range(500))
        xml = (
            '<?xml version="1.0"?>'
            f'<topic id="t"><title>T</title><body><p>{words}</p></body></topic>'
        )
        chunks = chunk_dita(xml, "docs/long.dita", "dita", cfg)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.token_count <= cfg.chunking.max_tokens
            assert chunk.metadata.get("truncated") is True or chunk.token_count <= 30


# ---------------------------------------------------------------------------
# Entity expansion safety
# ---------------------------------------------------------------------------

class TestEntityExpansionSafety:
    """Test that XML with entity declarations is rejected but DOCTYPE is allowed."""

    def test_doctype_allowed(self, default_config):
        xml = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE topic SYSTEM "topic.dtd">'
            '<topic id="t"><title>DOCTYPE Topic</title>'
            '<body><p>This topic has a standard DOCTYPE declaration which is normal '
            'for DITA files and should be processed without any issues by the chunker.</p></body></topic>'
        )
        chunks = chunk_dita(xml, "docs/dtd.dita", "dita", default_config)
        assert len(chunks) == 1
        assert chunks[0].content is not None

    def test_entity_declaration_rejected(self, default_config):
        xml = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE foo [<!ENTITY xxe "expanded">]>'
            '<topic id="t"><title>T</title><body><p>&xxe;</p></body></topic>'
        )
        chunks = chunk_dita(xml, "docs/xxe.dita", "dita", default_config)
        assert chunks == []
