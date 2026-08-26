"""Tests for the YAML and Helm template chunkers.

Covers the two things the generic blank-line fallback could not do:
splitting YAML along its own indentation, and splitting a Helm ``.tpl``
library into its named definitions.
"""

import pytest

from lib.chunkers.common import count_tokens, detect_language
from lib.chunkers.helm_template import chunk_helm_template, _find_definitions
from lib.chunkers.yaml_config import MAX_SPLIT_DEPTH, chunk_yaml
from lib.chunkers.yaml_structure import entries, entry_key, split_documents
from lib.models import ChunkType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Deliberately blank-line free, which is what the fallback could not handle.
DEPLOYMENT_YAML = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: privacy-service
  labels:
    app: privacy-service
    release: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: privacy-service
  template:
    spec:
      containers:
        - name: privacy-service
          image: registry.example.com/privacy-service:1.4.0
          ports:
            - containerPort: 8443
              protocol: TCP
          env:
            - name: LOG_LEVEL
              value: info
            - name: DB_HOST
              value: postgres.internal
            - name: DB_PORT
              value: "5432"
            - name: TLS_ENABLED
              value: "true"
"""

# A Helm manifest: template directives sit flush left while standing for
# content nested several levels deep.
HELM_MANIFEST_YAML = """\
apiVersion: v1
kind: Service
metadata:
  name: {{ include "chart.name" . }}
  labels:
{{- include "chart.labels" . | nindent 4 }}
  annotations:
{{- include "chart.annotations" . | nindent 4 }}
spec:
  type: ClusterIP
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
"""

HELM_TPL = """\
{{/* vim: set filetype=mustache: */}}

{{- define "chart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "chart.scheme" -}}
{{- if .Values.tls.enabled }}
{{- print "https" }}
{{- else }}
{{- print "http" }}
{{- end }}
{{- end -}}

{{- define "chart.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 -}}
{{- end -}}
"""


# ---------------------------------------------------------------------------
# YAML structure detection
# ---------------------------------------------------------------------------

class TestYamlStructure:
    """The primitives the recursive splitter is built on."""

    @pytest.mark.parametrize("line,expected", [
        ("name: privacy-service", "name"),
        ("spec:", "spec"),
        ('"quoted key": 1', "quoted key"),
        ("- name: proxy", "-"),
        ("-", "-"),
        ("{{- if .Values.tls.enabled }}", None),
        ("plain text with no colon", None),
    ])
    def testentry_key(self, line: str, expected) -> None:
        assert entry_key(line.strip()) == expected

    def test_top_levelentries(self) -> None:
        lines = DEPLOYMENT_YAML.split("\n")
        keys = [e.key for e in entries(lines, 0, len(lines))]
        assert keys == ["apiVersion", "kind", "metadata", "spec"]

    def test_template_lines_do_not_flatten_the_structure(self) -> None:
        """A flush-left {{- include }} must not be read as top-level.

        Helm writes those at column 0 whatever depth they stand for. Taking
        their indentation at face value makes every nested block look
        top-level, and the split collapses to a token-boundary cut.
        """
        lines = HELM_MANIFEST_YAML.split("\n")
        keys = [e.key for e in entries(lines, 0, len(lines))]
        assert keys == ["apiVersion", "kind", "metadata", "spec"]

    def test_comments_attach_to_the_key_below(self) -> None:
        lines = "# how many replicas\nreplicas: 3\nimage: nginx\n".split("\n")
        found = entries(lines, 0, len(lines))
        assert [e.key for e in found] == ["replicas", "image"]
        assert found[0].start == 0, "the comment belongs to replicas"

    @pytest.mark.parametrize("text,expected", [
        ("a: 1\nb: 2\n", 1),
        ("a: 1\n---\nb: 2\n", 2),
        ("---\na: 1\n---\nb: 2\n", 2),
        ("a: 1\n--- # a comment\nb: 2\n", 2),
    ])
    def test_document_splitting(self, text: str, expected: int) -> None:
        assert len(split_documents(text.split("\n"))) == expected


# ---------------------------------------------------------------------------
# YAML chunking
# ---------------------------------------------------------------------------

class TestYamlChunking:
    """End-to-end behaviour of the YAML chunker."""

    def test_small_file_is_one_chunk(self, default_config) -> None:
        text = "name: privacy-service\nreplicas: 3\nimage: nginx:1.25\nport: 8443\n"
        chunks = chunk_yaml(text, "values.yaml", "yaml", default_config)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.CONFIG_BLOCK
        assert chunks[0].metadata["key_path"] == []

    def test_oversized_file_splits_on_keys(self, small_config) -> None:
        chunks = chunk_yaml(DEPLOYMENT_YAML, "deployment.yaml", "yaml", small_config)
        assert len(chunks) > 1
        paths = [c.symbol_name for c in chunks if c.symbol_name]
        assert any(p.startswith("spec") for p in paths), paths

    def test_nested_chunks_carry_their_key_path(self, small_config) -> None:
        chunks = chunk_yaml(DEPLOYMENT_YAML, "deployment.yaml", "yaml", small_config)
        nested = [c for c in chunks if c.metadata["key_path"]]
        assert nested, "an oversized file must produce nested chunks"
        for chunk in nested:
            assert chunk.content.startswith("# "), "nested chunks need a breadcrumb"
            assert chunk.metadata["key_path"][0] in ("metadata", "spec")

    def test_sequence_steps_render_as_brackets(self, small_config) -> None:
        chunks = chunk_yaml(DEPLOYMENT_YAML, "deployment.yaml", "yaml", small_config)
        paths = [c.symbol_name or "" for c in chunks]
        assert any("[]" in p for p in paths), paths
        assert not any("- name" in p for p in paths), paths

    def test_no_chunk_exceeds_the_budget(self, small_config) -> None:
        chunks = chunk_yaml(DEPLOYMENT_YAML, "deployment.yaml", "yaml", small_config)
        for chunk in chunks:
            assert count_tokens(chunk.content) <= small_config.chunking.max_tokens

    def test_line_numbers_are_ordered_and_in_range(self, small_config) -> None:
        total = DEPLOYMENT_YAML.count("\n") + 1
        for chunk in chunk_yaml(DEPLOYMENT_YAML, "deployment.yaml", "yaml", small_config):
            assert 1 <= chunk.start_line <= total
            assert chunk.end_line >= chunk.start_line

    def test_documents_are_numbered(self, default_config) -> None:
        service = (
            "kind: Service\nmetadata:\n  name: privacy-service\n"
            "spec:\n  type: ClusterIP\n  ports:\n    - port: 8443\n"
        )
        deployment = (
            "kind: Deployment\nmetadata:\n  name: privacy-service\n"
            "spec:\n  replicas: 3\n  revisionHistoryLimit: 2\n"
        )
        chunks = chunk_yaml(
            f"{service}---\n{deployment}", "multi.yaml", "yaml", default_config,
        )
        assert [c.metadata["doc_index"] for c in chunks] == [0, 1]

    def test_single_document_has_no_doc_index(self, default_config) -> None:
        text = (
            "kind: Service\nmetadata:\n  name: privacy-service\n"
            "spec:\n  type: ClusterIP\n  ports:\n    - port: 8443\n"
        )
        chunks = chunk_yaml(text, "one.yaml", "yaml", default_config)
        assert chunks and "doc_index" not in chunks[0].metadata

    def test_helm_manifest_splits_by_key(self, small_config) -> None:
        chunks = chunk_yaml(HELM_MANIFEST_YAML, "service.yaml", "yaml", small_config)
        assert len(chunks) > 1
        for chunk in chunks:
            assert count_tokens(chunk.content) <= small_config.chunking.max_tokens

    def test_block_scalar_is_split_without_crashing(self, small_config) -> None:
        """Embedded XML in a block scalar has no YAML structure to use."""
        blob = "\n".join(f'    <kpi name="metric_{i}" aggregation="sum"/>' for i in range(200))
        text = f"config:\n  aggregator: |\n{blob}\n"
        chunks = chunk_yaml(text, "aggregator-values.yaml", "yaml", small_config)
        assert chunks
        for chunk in chunks:
            assert count_tokens(chunk.content) <= small_config.chunking.max_tokens

    def test_deep_nesting_terminates(self, small_config) -> None:
        """A file nested past MAX_SPLIT_DEPTH must still finish."""
        depth = MAX_SPLIT_DEPTH + 6
        lines = [("  " * i) + f"level{i}:" for i in range(depth)]
        lines.append(("  " * depth) + "value: " + ("x " * 400))
        chunks = chunk_yaml("\n".join(lines), "deep.yaml", "yaml", small_config)
        assert chunks

    def test_empty_content_yields_nothing(self, default_config) -> None:
        assert chunk_yaml("\n\n  \n", "empty.yaml", "yaml", default_config) == []

    @pytest.mark.parametrize("path,expected", [
        ("chart/values.yaml", "yaml"),
        ("ci/pipeline.yml", "yaml"),
        ("chart/templates/_helpers.tpl", "helm"),
    ])
    def test_extensions_route_here(self, path: str, expected: str) -> None:
        assert detect_language(path) == expected


# ---------------------------------------------------------------------------
# Helm template libraries
# ---------------------------------------------------------------------------

class TestHelmTemplateChunking:
    """One chunk per named definition, with the name attached."""

    def test_each_definition_becomes_a_chunk(self, default_config) -> None:
        chunks = chunk_helm_template(HELM_TPL, "_helpers.tpl", "helm", default_config)
        names = [c.symbol_name for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert names == ["chart.name", "chart.scheme", "chart.fullname"]

    def test_definition_name_reaches_metadata(self, default_config) -> None:
        chunks = chunk_helm_template(HELM_TPL, "_helpers.tpl", "helm", default_config)
        named = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert all(c.metadata["template_name"] == c.symbol_name for c in named)

    def test_nested_if_does_not_close_the_definition(self, default_config) -> None:
        """The {{ end }} of an inner conditional must not end the define."""
        definitions = _find_definitions(HELM_TPL.split("\n"))
        scheme = next(d for d in definitions if d.name == "chart.scheme")
        body = "\n".join(HELM_TPL.split("\n")[scheme.start:scheme.end])
        assert '{{- print "https" }}' in body
        assert '{{- print "http" }}' in body

    def test_definitions_do_not_overlap(self, default_config) -> None:
        definitions = _find_definitions(HELM_TPL.split("\n"))
        for earlier, later in zip(definitions, definitions[1:]):
            assert earlier.end <= later.start

    def test_text_outside_definitions_is_kept(self, default_config) -> None:
        text = "{{/* a long header comment explaining what this library does */}}\n" * 4
        chunks = chunk_helm_template(
            text + HELM_TPL, "_helpers.tpl", "helm", default_config,
        )
        module_level = [c for c in chunks if c.chunk_type == ChunkType.MODULE_LEVEL]
        assert module_level
        assert "header comment" in module_level[0].content

    def test_unterminated_definition_runs_to_end_of_file(self, default_config) -> None:
        text = '{{- define "chart.broken" -}}\nvalue: {{ .Values.x }}\n'
        definitions = _find_definitions(text.split("\n"))
        assert len(definitions) == 1
        assert definitions[0].end == len(text.split("\n"))

    def test_file_without_definitions_still_chunks(self, default_config) -> None:
        text = "just some text\n" * 30
        chunks = chunk_helm_template(text, "notes.tpl", "helm", default_config)
        assert chunks
        assert all(c.chunk_type == ChunkType.MODULE_LEVEL for c in chunks)

    def test_oversized_definition_is_split(self, small_config) -> None:
        body = "\n".join(f'  key{i}: {{{{ .Values.setting{i} }}}}' for i in range(120))
        text = f'{{{{- define "chart.big" -}}}}\n{body}\n{{{{- end -}}}}\n'
        chunks = chunk_helm_template(text, "_helpers.tpl", "helm", small_config)
        assert len(chunks) > 1
        for chunk in chunks:
            assert count_tokens(chunk.content) <= small_config.chunking.max_tokens

    def test_empty_file_yields_nothing(self, default_config) -> None:
        assert chunk_helm_template("\n\n", "empty.tpl", "helm", default_config) == []
