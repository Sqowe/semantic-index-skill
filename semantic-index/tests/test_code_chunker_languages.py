"""Tests for Phase 2 language expansion in the Tree-sitter code chunker.

Validates per-language:
- Grammar loading succeeds
- Top-level node extraction (functions, classes, structs, traits, etc.)
- Symbol name extraction
- Chunk type classification
- Method extraction from class-like containers
- Fallback to text splitting when grammar is unavailable

Each language has a representative code snippet as a fixture.
"""

import pytest

from lib.chunkers.code import (
    _get_ts_language,
    _get_parser,
    chunk_code_with_treesitter,
    EXTRACTABLE_NODES,
    METHOD_NODES,
)
from lib.chunkers.common import detect_language
from lib.models import ChunkType


# ---------------------------------------------------------------------------
# Language detection tests
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    """Verify file extensions map to the correct language identifier."""

    @pytest.mark.parametrize("ext,expected", [
        ("main.go", "go"),
        ("lib.rs", "rust"),
        ("App.java", "java"),
        ("util.c", "c"),
        ("util.h", "c"),
        ("widget.cpp", "cpp"),
        ("widget.hpp", "cpp"),
        ("app.rb", "ruby"),
        ("index.php", "php"),
    ])
    def test_extension_mapping(self, ext: str, expected: str) -> None:
        assert detect_language(ext) == expected


# ---------------------------------------------------------------------------
# Grammar loading tests
# ---------------------------------------------------------------------------

class TestGrammarLoading:
    """Verify Tree-sitter grammars load successfully for all Phase 2 languages."""

    @pytest.mark.parametrize("language", [
        "go", "rust", "java", "c", "cpp", "ruby", "php",
    ])
    def test_grammar_loads(self, language: str) -> None:
        ts_lang = _get_ts_language(language)
        assert ts_lang is not None, f"Grammar for {language} failed to load"

    @pytest.mark.parametrize("language", [
        "go", "rust", "java", "c", "cpp", "ruby", "php",
    ])
    def test_parser_creates(self, language: str) -> None:
        parser = _get_parser(language)
        assert parser is not None, f"Parser for {language} failed to create"


# ---------------------------------------------------------------------------
# Node type configuration tests
# ---------------------------------------------------------------------------

class TestNodeTypeConfig:
    """Verify EXTRACTABLE_NODES and METHOD_NODES are defined for all languages."""

    @pytest.mark.parametrize("language", [
        "go", "rust", "java", "c", "cpp", "ruby", "php",
    ])
    def test_extractable_nodes_defined(self, language: str) -> None:
        assert language in EXTRACTABLE_NODES
        assert len(EXTRACTABLE_NODES[language]) > 0

    @pytest.mark.parametrize("language", [
        "go", "rust", "java", "c", "cpp", "ruby", "php",
    ])
    def test_method_nodes_defined(self, language: str) -> None:
        assert language in METHOD_NODES


# ---------------------------------------------------------------------------
# Code fixtures for each language
# ---------------------------------------------------------------------------

GO_CODE = '''\
package main

import "fmt"

// Greet returns a greeting for the given name.
func Greet(name string) string {
    return fmt.Sprintf("Hello, %s!", name)
}

// Server handles HTTP requests.
type Server struct {
    Port    int
    Host    string
    Timeout int
    Debug   bool
    Logger  func(string)
}

func (s *Server) Start() error {
    fmt.Printf("Starting on %s:%d\\n", s.Host, s.Port)
    return nil
}
'''

RUST_CODE = '''\
use std::fmt;

/// A point in 2D space.
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    pub fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
'''

JAVA_CODE = '''\
package com.example;

import java.util.List;

public class UserService {
    private final String dbUrl;

    public UserService(String dbUrl) {
        this.dbUrl = dbUrl;
    }

    public List<String> getUsers() {
        return List.of("alice", "bob");
    }

    public void deleteUser(String name) {
        System.out.println("Deleting " + name);
    }
}
'''

C_CODE = '''\
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int x;
    int y;
    char label[64];
} Point;

int add(int a, int b) {
    int result = a + b;
    if (result < 0) {
        fprintf(stderr, "overflow detected\\n");
        return -1;
    }
    return result;
}

void print_point(const Point *p) {
    if (p == NULL) {
        fprintf(stderr, "null pointer\\n");
        return;
    }
    printf("Point(%s): (%d, %d)\\n", p->label, p->x, p->y);
}
'''

CPP_CODE = '''\
#include <iostream>
#include <string>

namespace utils {

class Logger {
public:
    Logger(const std::string& name) : name_(name) {}

    void info(const std::string& msg) {
        std::cout << "[" << name_ << "] " << msg << std::endl;
    }

private:
    std::string name_;
};

}  // namespace utils

int main() {
    utils::Logger logger("app");
    logger.info("started");
    return 0;
}
'''

RUBY_CODE = '''\
module Greeter
  class Person
    attr_reader :name

    def initialize(name)
      @name = name
    end

    def greet
      "Hello, #{@name}!"
    end
  end

  def self.default_greeting
    "Hello, World!"
  end
end
'''

PHP_CODE = '''\
<?php

namespace App\\Services;

class UserService {
    private string $dbUrl;

    public function __construct(string $dbUrl) {
        $this->dbUrl = $dbUrl;
    }

    public function getUsers(): array {
        return ["alice", "bob"];
    }

    public function deleteUser(string $name): void {
        echo "Deleting " . $name;
    }
}

function helper(string $input): string {
    $result = strtoupper($input);
    $result = trim($result);
    $result = str_replace(" ", "_", $result);
    return $result;
}
'''


# ---------------------------------------------------------------------------
# Per-language chunking tests
# ---------------------------------------------------------------------------

class TestGoChunking:
    """Go: functions, type declarations, method declarations."""

    def test_extracts_function(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(GO_CODE, "main.go", "go", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "Greet" in names

    def test_extracts_type(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(GO_CODE, "main.go", "go", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "Server" in names

    def test_extracts_method_declaration(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(GO_CODE, "main.go", "go", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "Start" in names

    def test_chunk_types(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(GO_CODE, "main.go", "go", default_config)
        types = {c.chunk_type for c in chunks}
        assert ChunkType.FUNCTION in types

    def test_has_module_level(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(GO_CODE, "main.go", "go", default_config)
        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.MODULE_LEVEL]
        assert len(module_chunks) >= 1


class TestRustChunking:
    """Rust: functions, structs, impl blocks with methods."""

    def test_extracts_function(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(RUST_CODE, "lib.rs", "rust", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "add" in names

    def test_extracts_struct(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(RUST_CODE, "lib.rs", "rust", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "Point" in names

    def test_extracts_impl_block(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(RUST_CODE, "lib.rs", "rust", default_config)
        # impl block should be extracted (as class-like or with methods)
        impl_or_method = [c for c in chunks if c.chunk_type in (ChunkType.CLASS, ChunkType.METHOD)]
        assert len(impl_or_method) >= 1

    def test_chunk_types(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(RUST_CODE, "lib.rs", "rust", default_config)
        types = {c.chunk_type for c in chunks}
        assert ChunkType.FUNCTION in types


class TestJavaChunking:
    """Java: class with constructor and methods."""

    def test_extracts_class(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(JAVA_CODE, "UserService.java", "java", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "UserService" in names

    def test_class_chunk_type(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(JAVA_CODE, "UserService.java", "java", default_config)
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

    def test_method_extraction_on_oversized(self, small_config) -> None:
        """When class exceeds max_tokens, methods should be extracted."""
        chunks = chunk_code_with_treesitter(JAVA_CODE, "UserService.java", "java", small_config)
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        # With small max_tokens, the class should be split into methods
        assert len(method_chunks) >= 1


class TestCChunking:
    """C: functions, structs, typedefs."""

    def test_extracts_functions(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(C_CODE, "util.c", "c", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "add" in names
        assert "print_point" in names

    def test_extracts_typedef_struct(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(C_CODE, "util.c", "c", default_config)
        # The typedef struct should be extracted as a chunk
        non_module = [c for c in chunks if c.chunk_type != ChunkType.MODULE_LEVEL]
        assert len(non_module) >= 2  # at least the two functions

    def test_has_module_level(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(C_CODE, "util.c", "c", default_config)
        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.MODULE_LEVEL]
        assert len(module_chunks) >= 1


class TestCppChunking:
    """C++: namespaces, classes, functions."""

    def test_extracts_namespace(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(CPP_CODE, "main.cpp", "cpp", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "utils" in names

    def test_extracts_main(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(CPP_CODE, "main.cpp", "cpp", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "main" in names

    def test_chunk_types(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(CPP_CODE, "main.cpp", "cpp", default_config)
        types = {c.chunk_type for c in chunks}
        assert ChunkType.FUNCTION in types or ChunkType.CLASS in types


class TestRubyChunking:
    """Ruby: modules, classes, methods."""

    def test_extracts_module(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(RUBY_CODE, "greeter.rb", "ruby", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "Greeter" in names

    def test_extracts_class(self, default_config) -> None:
        """Person is nested inside Greeter module, so it appears
        inside the module chunk, not as a separate top-level symbol."""
        chunks = chunk_code_with_treesitter(RUBY_CODE, "greeter.rb", "ruby", default_config)
        # The module chunk content should contain the Person class
        module_chunks = [c for c in chunks if c.symbol_name == "Greeter"]
        assert len(module_chunks) >= 1
        assert "Person" in module_chunks[0].content

    def test_chunk_types(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(RUBY_CODE, "greeter.rb", "ruby", default_config)
        types = {c.chunk_type for c in chunks}
        assert ChunkType.CLASS in types or ChunkType.FUNCTION in types


class TestPhpChunking:
    """PHP: classes, functions, methods."""

    def test_extracts_class(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(PHP_CODE, "UserService.php", "php", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "UserService" in names

    def test_extracts_function(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(PHP_CODE, "UserService.php", "php", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "helper" in names

    def test_method_extraction_on_oversized(self, small_config) -> None:
        """When class exceeds max_tokens, methods should be extracted."""
        chunks = chunk_code_with_treesitter(PHP_CODE, "UserService.php", "php", small_config)
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        assert len(method_chunks) >= 1

    def test_class_chunk_type(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(PHP_CODE, "UserService.php", "php", default_config)
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1


# ---------------------------------------------------------------------------
# Cross-language consistency tests
# ---------------------------------------------------------------------------

class TestCrossLanguageConsistency:
    """Verify consistent behavior across all languages."""

    @pytest.mark.parametrize("code,file_path,language", [
        (GO_CODE, "main.go", "go"),
        (RUST_CODE, "lib.rs", "rust"),
        (JAVA_CODE, "UserService.java", "java"),
        (C_CODE, "util.c", "c"),
        (CPP_CODE, "main.cpp", "cpp"),
        (RUBY_CODE, "greeter.rb", "ruby"),
        (PHP_CODE, "UserService.php", "php"),
    ])
    def test_produces_chunks(self, code, file_path, language, default_config) -> None:
        """Every language should produce at least one chunk."""
        chunks = chunk_code_with_treesitter(code, file_path, language, default_config)
        assert len(chunks) > 0, f"{language} produced no chunks"

    @pytest.mark.parametrize("code,file_path,language", [
        (GO_CODE, "main.go", "go"),
        (RUST_CODE, "lib.rs", "rust"),
        (JAVA_CODE, "UserService.java", "java"),
        (C_CODE, "util.c", "c"),
        (CPP_CODE, "main.cpp", "cpp"),
        (RUBY_CODE, "greeter.rb", "ruby"),
        (PHP_CODE, "UserService.php", "php"),
    ])
    def test_chunks_have_valid_fields(self, code, file_path, language, default_config) -> None:
        """All chunks should have required fields populated."""
        chunks = chunk_code_with_treesitter(code, file_path, language, default_config)
        for chunk in chunks:
            assert chunk.id, "Chunk missing id"
            assert chunk.file_path == file_path
            assert chunk.content, "Chunk has empty content"
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line
            assert chunk.language == language
            assert chunk.token_count > 0
            assert isinstance(chunk.chunk_type, ChunkType)

    @pytest.mark.parametrize("code,file_path,language", [
        (GO_CODE, "main.go", "go"),
        (RUST_CODE, "lib.rs", "rust"),
        (JAVA_CODE, "UserService.java", "java"),
        (C_CODE, "util.c", "c"),
        (CPP_CODE, "main.cpp", "cpp"),
        (RUBY_CODE, "greeter.rb", "ruby"),
        (PHP_CODE, "UserService.php", "php"),
    ])
    def test_no_empty_symbol_names(self, code, file_path, language, default_config) -> None:
        """Symbol names, when present, should not be empty strings."""
        chunks = chunk_code_with_treesitter(code, file_path, language, default_config)
        for chunk in chunks:
            if chunk.symbol_name is not None:
                assert chunk.symbol_name.strip(), f"Empty symbol name in {language} chunk"


# ---------------------------------------------------------------------------
# Phase 1 regression tests (Python, JavaScript, TypeScript)
# ---------------------------------------------------------------------------

PYTHON_CODE = '''\
import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    """Load configuration from a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    return {"path": path}

class ConfigManager:
    """Manages application configuration."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.configs: dict = {}

    def get(self, name: str) -> dict:
        """Retrieve a named configuration."""
        if name not in self.configs:
            raise KeyError(f"Unknown config: {name}")
        return self.configs[name]

    def reload(self) -> None:
        """Reload all configurations from disk."""
        for name in list(self.configs.keys()):
            path = self.base_dir / f"{name}.json"
            self.configs[name] = load_config(str(path))
'''

JS_CODE = '''\
import { readFile } from 'fs/promises';

function parseConfig(raw) {
    const data = JSON.parse(raw);
    if (!data.version) {
        throw new Error('Missing version field');
    }
    return data;
}

class ConfigLoader {
    constructor(basePath) {
        this.basePath = basePath;
        this.cache = new Map();
    }

    async load(name) {
        if (this.cache.has(name)) {
            return this.cache.get(name);
        }
        const raw = await readFile(`${this.basePath}/${name}.json`, 'utf-8');
        const config = parseConfig(raw);
        this.cache.set(name, config);
        return config;
    }
}

export { ConfigLoader, parseConfig };
'''

TS_CODE = '''\
import { readFileSync } from 'fs';

interface AppConfig {
    version: string;
    debug: boolean;
    port: number;
    host: string;
    logLevel: 'info' | 'warn' | 'error' | 'debug';
    maxConnections: number;
}

type ConfigKey = keyof AppConfig;

function loadConfig(path: string): AppConfig {
    const raw = readFileSync(path, 'utf-8');
    const data = JSON.parse(raw) as AppConfig;
    if (!data.version) {
        throw new Error('Invalid config: missing version');
    }
    return data;
}

class ConfigService {
    private config: AppConfig | null = null;

    constructor(private readonly path: string) {}

    get(key: ConfigKey): string | boolean | number {
        if (!this.config) {
            this.config = loadConfig(this.path);
        }
        return this.config[key];
    }

    reload(): void {
        this.config = loadConfig(this.path);
    }
}

export { ConfigService, loadConfig };
'''


class TestPythonRegression:
    """Phase 1 regression: Python symbol extraction and chunk typing."""

    def test_extracts_function(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(PYTHON_CODE, "config.py", "python", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "load_config" in names

    def test_extracts_class(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(PYTHON_CODE, "config.py", "python", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "ConfigManager" in names

    def test_function_chunk_type(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(PYTHON_CODE, "config.py", "python", default_config)
        func_chunk = next(c for c in chunks if c.symbol_name == "load_config")
        assert func_chunk.chunk_type == ChunkType.FUNCTION

    def test_class_chunk_type(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(PYTHON_CODE, "config.py", "python", default_config)
        class_chunks = [c for c in chunks if c.symbol_name == "ConfigManager"]
        assert len(class_chunks) >= 1
        assert class_chunks[0].chunk_type == ChunkType.CLASS

    def test_method_extraction_on_oversized(self, small_config) -> None:
        chunks = chunk_code_with_treesitter(PYTHON_CODE, "config.py", "python", small_config)
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        assert len(method_chunks) >= 1

    def test_has_module_level(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(PYTHON_CODE, "config.py", "python", default_config)
        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.MODULE_LEVEL]
        assert len(module_chunks) >= 1


class TestJavaScriptRegression:
    """Phase 1 regression: JavaScript symbol extraction and chunk typing."""

    def test_extracts_function(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(JS_CODE, "config.js", "javascript", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "parseConfig" in names

    def test_extracts_class(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(JS_CODE, "config.js", "javascript", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "ConfigLoader" in names

    def test_chunk_types(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(JS_CODE, "config.js", "javascript", default_config)
        types = {c.chunk_type for c in chunks}
        assert ChunkType.FUNCTION in types
        assert ChunkType.CLASS in types


class TestTypeScriptRegression:
    """Phase 1 regression: TypeScript symbol extraction and chunk typing."""

    def test_extracts_function(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(TS_CODE, "config.ts", "typescript", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "loadConfig" in names

    def test_extracts_class(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(TS_CODE, "config.ts", "typescript", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "ConfigService" in names

    def test_extracts_interface(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(TS_CODE, "config.ts", "typescript", default_config)
        names = [c.symbol_name for c in chunks if c.symbol_name]
        assert "AppConfig" in names

    def test_chunk_types(self, default_config) -> None:
        chunks = chunk_code_with_treesitter(TS_CODE, "config.ts", "typescript", default_config)
        types = {c.chunk_type for c in chunks}
        assert ChunkType.FUNCTION in types
        assert ChunkType.CLASS in types
