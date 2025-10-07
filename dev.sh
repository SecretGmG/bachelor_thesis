#!/usr/bin/env bash
set -e

source .venv/bin/activate
maturin develop -m code/numerical_code_solution/Cargo.toml
