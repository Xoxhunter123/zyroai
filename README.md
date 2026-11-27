# Zyro AI — Render 12 vCPU / 96 GB RAM Agentic Backend

Push **this `render/` folder** directly to a new GitHub repository, connect it to Render, and run **`Qwen3-Coder-30B-A3B-Instruct`** (19–35 tok/s, MoE 3B active) with full OpenCode-style file creation & terminal execution for the **Zyro AI** Android app.

## Files Included
- `Dockerfile` — Builds `llama.cpp` (`llama-server`) with AVX2/FMA + Python 3, Node.js, Git, Ripgrep, SQLite.
- `start_render.sh` — Downloads & caches `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (~18.6 GB) on `/data/models`, launches `llama-server` with `--jinja --ctx-size 32768 --threads 12`, and runs `server.py`.
- `server.py` — OpenCode-style Agent API (`/v1/agent/run`, `/v1/chat/completions`, `/v1/workspace/*`, `/health`).
- `render.yaml` — Ready-to-use Render Blueprint.

## Push to GitHub
```bash
cd render
git init
git add .
git commit -m "Zyro AI Render Agentic Backend"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/zyro-ai-render.git
git push -u origin main
```

## Exact Render Dashboard Settings
- **Service Type**: Web Service → **Runtime**: `Docker`
- **Instance Type**: Your **12 vCPU / 96 GB RAM** plan
- **Health Check Path**: `/health`
- **Persistent Disk**: Mount Path `/data`, Size `50 GB`
- **Environment Variables**:
  - `PORT` = `10000`
  - `DATA_DIR` = `/data`
  - `MODEL_REPO` = `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`
  - `MODEL_FILE` = `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
  - `MODEL_ALIAS` = `Qwen3-Coder-30B-A3B-Instruct`
  - `CTX_SIZE` = `32768`
  - `THREADS` = `12`
  - `ZYRO_API_KEY` = *(set any strong secret password, e.g. `zyro-super-secret-2026`)*
