#!/bin/bash
# MCP server launcher for mcp-raven
cd "$(dirname "$0")"
set -a
source .env
set +a
exec "$HOME/.local/bin/uv" run main.py
