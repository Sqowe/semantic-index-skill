# Supported Languages — Semantic Index

> Tree-sitter grammars used for AST-aware code chunking, plus file extension
> mappings and extractable node types per language.

---

## Overview

The semantic index uses Tree-sitter for AST-aware chunking of code files.
Each supported language has a dedicated grammar that parses source code into
a syntax tree, allowing the chunker to extract meaningful units (functions,
classes, methods) instead of splitting at arbitrary line boundaries.

Languages without a Tree-sitter grammar fall back to blank-line splitting.

---

## Supported Languages

### Python

| Attribute | Value |
|-----------|-------|
| Grammar package | `tree-sitter-python` |
| File extensions | `.py` |
| Top-level nodes | `function_definition`, `class_definition`, `decorated_definition` |
| Method nodes | `function_definition` (inside class body) |
| Body node types | `block` |

### JavaScript

| Attribute | Value |
|-----------|-------|
| Grammar package | `tree-sitter-javascript` |
| File extensions | `.js`, `.jsx` |
| Top-level nodes | `function_declaration`, `class_declaration`, `lexical_declaration`, `export_statement` |
| Method nodes | `method_definition` |
| Body node types | `class_body`, `statement_block` |

### TypeScript

| Attribute | Value |
|-----------|-------|
| Grammar package | `tree-sitter-typescript` |
| File extensions | `.ts`, `.tsx` |
| Top-level nodes | `function_declaration`, `class_declaration`, `lexical_declaration`, `export_statement`, `interface_declaration`, `type_alias_declaration` |
| Method nodes | `method_definition`, `public_field_definition` |
| Body node types | `class_body`, `statement_block` |

### Go

| Attribute | Value |
|-----------|-------|
| Grammar package | `tree-sitter-go` |
| File extensions | `.go` |
| Top-level nodes | `function_declaration`, `method_declaration`, `type_declaration` |
| Method nodes | None (Go methods are top-level `method_declaration` nodes) |
| Body node types | `block` |
| Notes | Go methods use receiver syntax (`func (s *Server) Handle()`) and are parsed as top-level `method_declaration` nodes, not nested inside a type. `type_declaration` covers structs, interfaces, and type aliases. |

### Rust

| Attribute | Value |
|-----------|-------|
| Grammar package | `tree-sitter-rust` |
| File extensions | `.rs` |
| Top-level nodes | `function_item`, `struct_item`, `enum_item`, `impl_item`, `trait_item`, `type_item`, `const_item`, `static_item`, `macro_definition` |
| Method nodes | `function_item` (inside `impl` / `trait` blocks) |
| Body node types | `declaration_list` |
| Notes | `impl_item` blocks are treated as class-like containers. Methods inside `impl` are extracted individually when the block exceeds `max_tokens`. |

### Java

| Attribute | Value |
|-----------|-------|
| Grammar package | `tree-sitter-java` |
| File extensions | `.java` |
| Top-level nodes | `class_declaration`, `interface_declaration`, `enum_declaration`, `record_declaration`, `annotation_type_declaration` |
| Method nodes | `method_declaration`, `constructor_declaration` |
| Body node types | `class_body`, `enum_body` |
| Notes | Java files typically have one public class per file. Inner classes are chunked as part of the outer class unless the outer class is oversized, in which case methods are extracted individually. |

### C

| Attribute | Value |
|-----------|-------|
| Grammar package | `tree-sitter-c` |
| File extensions | `.c`, `.h` |
| Top-level nodes | `function_definition`, `struct_specifier`, `enum_specifier`, `type_definition`, `declaration` |
| Method nodes | None (C has no classes) |
| Body node types | N/A |
| Notes | C has no class/method concept. All functions are top-level. `struct_specifier` and `enum_specifier` capture type definitions. `declaration` catches global variables and function prototypes. |

### C++

| Attribute | Value |
|-----------|-------|
| Grammar package | `tree-sitter-cpp` |
| File extensions | `.cpp`, `.hpp` |
| Top-level nodes | `function_definition`, `class_specifier`, `struct_specifier`, `enum_specifier`, `namespace_definition`, `template_declaration`, `type_definition`, `declaration` |
| Method nodes | `function_definition` (inside class body); `field_declaration` nodes containing a `function_declarator` are also recognized as methods during oversized class splitting |
| Body node types | `field_declaration_list`, `declaration_list` |
| Notes | C++ classes use `field_declaration_list` as their body node. `namespace_definition` is treated as a class-like container for method extraction. `template_declaration` wraps templated functions/classes. |

### Ruby

| Attribute | Value |
|-----------|-------|
| Grammar package | `tree-sitter-ruby` |
| File extensions | `.rb` |
| Top-level nodes | `method`, `class`, `module`, `singleton_method` |
| Method nodes | `method`, `singleton_method` |
| Body node types | `body` |
| Notes | Ruby `module` nodes are treated as class-like containers. `singleton_method` covers `self.method_name` class-level methods. |

### PHP

| Attribute | Value |
|-----------|-------|
| Grammar package | `tree-sitter-php` |
| File extensions | `.php` |
| Top-level nodes | `function_definition`, `class_declaration`, `interface_declaration`, `trait_declaration`, `enum_declaration` |
| Method nodes | `method_declaration` |
| Body node types | `declaration_list` |
| Notes | PHP traits are treated as class-like containers. The grammar exposes `language_php()` (some versions use `language()`); the loader handles both. |

---

## Fallback Behavior

Files with extensions not listed above (e.g., `.txt`, `.rst`, `.cfg`) use
blank-line-based splitting. The fallback:

1. Splits content at double-newline (`\n\n`) boundaries
2. Merges adjacent blocks until `max_tokens` is reached
3. Marks chunks as `ChunkType.UNKNOWN`

This produces reasonable chunks for prose and configuration files but lacks
the semantic precision of AST-aware splitting.

---

## DITA XML Documentation

DITA (Darwin Information Typing Architecture) files use XML-aware chunking
via Python's built-in `xml.etree.ElementTree`. No Tree-sitter grammar or
external dependencies are needed.

### Supported Topic Types

| Topic Type | Root Element | Body Element | Description |
|-----------|-------------|-------------|-------------|
| Generic topic | `<topic>` | `<body>` | General-purpose topic |
| Concept | `<concept>` | `<conbody>` | Conceptual/explanatory content |
| Task | `<task>` | `<taskbody>` | Step-by-step procedures |
| Reference | `<reference>` | `<refbody>` | Reference tables, APIs, specs |
| Glossary entry | `<glossentry>` | `<glossBody>` | Term definitions |
| Troubleshooting | `<troubleshooting>` | `<troublebody>` | Problem/solution pairs |
| Specializations | Any (with `class` attr) | Varies | Custom topic types extending `topic/topic` |

### Chunking Behavior

- Each topic = one chunk (title + shortdesc + body text, tags stripped)
- If a topic exceeds `chunk_max_tokens`, splits at `<section>` boundaries
- `<prolog>` metadata (keywords, audience, category) is prepended as context
- `xml:lang` attributes are propagated as chunk metadata
- `conref`/`conkeyref` attributes are noted in metadata (not resolved)

### DITA Map Files (.ditamap)

- `<topicref>` hierarchy is walked to extract navigation structure
- Produces a single "map overview" chunk with navtitle, href, and keys
- Useful for queries like "where is the installation guide?"

### Text Extraction Elements

Text is extracted from (in reading order):
`<title>`, `<shortdesc>`, `<abstract>`, `<p>`, `<li>`, `<sli>`, `<dt>`, `<dd>`,
`<note>`, `<section>`, `<example>`, `<step>`, `<cmd>`, `<info>`, `<stepresult>`,
`<result>`, `<prereq>`, `<context>`, `<codeblock>`, `<screen>`, `<msgblock>`,
`<entry>`, `<stentry>`.

### Skipped Elements

Structural/metadata-only elements are skipped:
`<prolog>` (extracted separately), `<related-links>`, `<link>`, `<topicmeta>`,
`<navref>`, `<anchor>`, `<data>`, `<data-about>`, `<foreign>`, `<unknown>`.

---

## Office Documents (PDF, DOCX, PPTX)

Office documents use binary-format extraction via dedicated Python libraries.
Text is extracted and chunked using format-aware strategies. Dependencies are
optional — install via `bash setup.sh --with-office`.

### Supported Formats

| Format | Library | Chunking Strategy | Chunk Type |
|--------|---------|-------------------|------------|
| PDF | `PyMuPDF>=1.24.0` (`fitz`) | Page-based splitting | `pdf_page` |
| DOCX | `python-docx>=1.1.0` | Heading-based sectioning | `docx_section` |
| PPTX | `python-pptx>=1.0.0` | Slide-based splitting | `pptx_slide` |

### PDF Chunking

- Text extracted per page via `page.get_text("text")`
- Document metadata (title, author) stored in chunk metadata
- Short consecutive pages are merged (below `min_tokens`)
- Long pages are split at paragraph boundaries (double newlines)
- `start_line` / `end_line` = page number (1-based)
- Scanned PDFs (image-only, no text layer) are skipped with a warning

### DOCX Chunking

- Paragraphs grouped by heading style (`Heading 1`, `Heading 2`, etc.)
- Each heading + its body paragraphs = one chunk (mirrors markdown chunker)
- Tables extracted as row-by-row text with ` | ` cell separators
- Document core properties (title, author) stored in chunk metadata
- `heading_path` metadata preserves the heading hierarchy
- `start_line` / `end_line` = 1-based section index

### PPTX Chunking

- Text extracted from all shapes: text frames, tables, grouped shapes
- Speaker notes included (prefixed with `[Speaker notes]`)
- Title placeholder text stored as `slide_title` metadata
- Image-only slides (no extractable text) are skipped
- Short consecutive slides are merged (below `min_tokens`)
- `start_line` / `end_line` = slide number (1-based)

### Known Limitations

- Scanned PDFs yield empty text (no OCR support)
- Complex table layouts are flattened to row-by-row text
- Embedded images, charts, and diagrams are invisible to text extraction
- Password-protected files cannot be opened (skipped with warning)
- Formatting (bold, italic, colors, fonts) is stripped — only raw text is indexed

---

## File Extension → Language Mapping

| Extension | Language | Chunking Strategy |
|-----------|----------|-------------------|
| `.py` | Python | Tree-sitter AST |
| `.js`, `.jsx` | JavaScript | Tree-sitter AST |
| `.ts`, `.tsx` | TypeScript | Tree-sitter AST |
| `.go` | Go | Tree-sitter AST |
| `.rs` | Rust | Tree-sitter AST |
| `.java` | Java | Tree-sitter AST |
| `.c`, `.h` | C | Tree-sitter AST |
| `.cpp`, `.hpp` | C++ | Tree-sitter AST |
| `.rb` | Ruby | Tree-sitter AST |
| `.php` | PHP | Tree-sitter AST |
| `.md`, `.mdx` | Markdown | Header-based |
| `.dita` | DITA XML | XML topic-based |
| `.ditamap` | DITA Map | XML topicref-based |
| `.pdf` | PDF | Page-based extraction |
| `.docx` | Word | Heading-based extraction |
| `.pptx` | PowerPoint | Slide-based extraction |
| `.txt`, `.rst` | Text | Blank-line fallback |

---

## Adding a New Language

To add Tree-sitter support for a new language:

1. Add the `tree-sitter-<lang>` package to `requirements.txt`
2. Add a grammar loading branch in `chunkers/code.py` → `_get_ts_language()`
3. Add entries to `EXTRACTABLE_NODES` and `METHOD_NODES` dicts
4. Add the file extension mapping in `chunkers/common.py` → `detect_language()`
5. Add the language to `TREESITTER_LANGUAGES` in `chunker.py`
6. Update this document
