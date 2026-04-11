---
name: Use uv sync instead of uv pip
description: Prefer uv sync --extra <name> over uv pip install for managing dependencies in this project
type: feedback
---
Use `uv sync --extra <name>` instead of `uv pip install -e ".[<name>]"` when installing dependencies.

**Why:** User preference — `uv sync` is the correct uv workflow for managing project environments declaratively.

**How to apply:** Whenever writing commands for this project that install dependencies, use:
- `uv sync` — base deps
- `uv sync --extra integration` — with integration extras
- `uv sync --extra quantization` — with quantization extras
- `uv sync --extra integration --extra quantization` — multiple extras

Never suggest `uv pip install -e ".[...]"` for this project.
