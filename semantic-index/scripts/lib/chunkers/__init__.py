"""Chunking strategy subpackage.

Each module implements a specific chunking strategy:
- code.py: Tree-sitter AST-aware code chunking
  Supported: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP
- markdown.py: Header-based markdown chunking
- dita.py: XML-aware DITA topic chunking (.dita, .ditamap)

The parent chunker module (lib/chunker.py) handles dispatch and shared utilities.
"""
