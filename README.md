# Versa

Versa is a **Modal-like** runner and MCP server for running workloads across backends:

- **Remote Jupyter** (e.g. Kaggle `/proxy`) via Jupyter REST + `jupyter_kernel_client`
- **Modal** (local `modal run ...` wrapper) when available

## Quickstart (CLI)

Create a venv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run a command on a remote Jupyter server:

```bash
python -m versa run --url "$REMOTE_JUPYTER_URL" --detach python -c "print('hello from versa')"
```

Run a Modal-style file on remote Jupyter (no modifications required):

```bash
python -m versa run --url "$REMOTE_JUPYTER_URL" \
  external/TPU-dlm/modal/benchmark_wedlm.py::run_kv_growth_benchmark_fa3 -- \
  --kv-lens-csv 4096,8192
```

## MCP

Run the Versa MCP server:

```bash
python mcp_servers/versa_mcp_server.py --host 127.0.0.1 --port 8010
```

Endpoint: `http://127.0.0.1:8010/mcp`

Docs:
- `mcp_servers/VERSA.md`
- `mcp_servers/REMOTE_JUPYTER_MCP.md`

