"""Chunking strategy subpackage.

Each module implements a specific chunking strategy:
- code.py: Tree-sitter AST-aware code chunking
- markdown.py: Header-based markdown chunking

The parent chunker module (lib/chunker.py) handles dispatch and shared utilities.
"""
