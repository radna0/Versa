# Versa

Versa is a **Modal-like** runner that can dispatch the same “run + logs + tail + kill + collect” workflow across backends:

- **`modal` backend**: runs `modal run ...` locally (Modal cloud does the compute)
- **`jupyter` backend**: runs on a remote Jupyter server (e.g. Kaggle `/proxy`)

This repo provides:

- A Versa MCP server: `mcp_servers/versa_mcp_server.py`
- A simple Versa CLI: `python -m versa ...`

## MCP Server

Run:

```bash
python mcp_servers/versa_mcp_server.py --host 127.0.0.1 --port 8010
```

MCP endpoint: `http://127.0.0.1:8010/mcp`

Tools:

- `versa_run`
- `versa_status`
- `versa_tail`
- `versa_kill`
- `versa_collect`

### `versa_run` (jupyter backend)

Key args:

- `backend="jupyter"`
- `url="https://.../proxy"`
- `token=""` (optional)
- either `command="..."` or `script_path="local/path.py"`
- or a Modal-style python file target (e.g. `external/TPU-dlm/modal/benchmark_wedlm.py::run_kv_growth_benchmark_fa3`)
- optional `sync_local_dir`, `sync_remote_dir` + `bootstrap_commands`

## CLI

Modal backend:

```bash
python -m versa run modal/benchmark_wedlm.py::run_kv_growth_benchmark_fa3 -- --kv-lens-csv 4096,8192
```

Jupyter backend: run a Modal-style file (no modifications to the file required):

```bash
python -m versa run --url "$REMOTE_JUPYTER_URL" \
  external/TPU-dlm/modal/benchmark_wedlm.py::run_kv_growth_benchmark_fa3 -- \
  --kv-lens-csv 4096,8192
```

Jupyter backend (Kaggle-style):

```bash
python -m versa run --url "$REMOTE_JUPYTER_URL" \\
  --bootstrap-cmd "python -V" \\
  --env WEDLM_FLASH_ATTN_BACKEND=fa3 \\
  "python -c \"import subprocess; print(subprocess.check_output(['nvidia-smi','-L']).decode())\""
```
