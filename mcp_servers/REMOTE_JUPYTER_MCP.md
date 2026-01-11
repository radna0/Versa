# Remote Jupyter MCP (Bridge)

This is a small **local MCP server** that exposes tools for controlling a **remote Jupyter Server** (e.g. Kaggle `/proxy`, local `jupyter lab`, JupyterHub) so other agents can run GPU workloads, sync code, and fetch logs/artifacts.

## Run

```bash
export REMOTE_JUPYTER_URL='https://.../proxy'   # or http://localhost:8888
export REMOTE_JUPYTER_TOKEN=''                  # optional
python mcp_servers/remote_jupyter_mcp_server.py --host 127.0.0.1 --port 8009
```

The MCP endpoint is `http://127.0.0.1:8009/mcp`.

## Connect (example: jupyter-ai-agents)

```bash
jupyter-ai-agents repl --mcp-servers http://127.0.0.1:8009/mcp
```

## Tools (current)

- `jupyter_ping`, `jupyter_env_snapshot`, `jupyter_list_kernels`, `jupyter_list_sessions`
- `jupyter_start_kernel`, `jupyter_shutdown_kernel`
- `jupyter_exec_python`
- `jupyter_list_dir`, `jupyter_read_text`, `jupyter_write_text`, `jupyter_upload_base64`
- `jupyter_upload_local_file`, `jupyter_download_base64`, `jupyter_download_to_local`, `jupyter_download_dir_to_local`
- `jupyter_sync_dir` (tar.gz sync; Modal-like `add_local_dir(copy=True)`)
- `jupyter_run_commands` (Modal-like `run_commands`)
- `jupyter_job_start`, `jupyter_job_status`, `jupyter_job_tail`, `jupyter_job_kill`
- `jupyter_job_submit`, `jupyter_job_collect` (declarative spec + artifact pull)

## Example: Modal-like job spec

Submit (via `jupyter_job_submit`):

```json
{
  "sync": [
    {
      "local_dir": "external/TPU-dlm",
      "remote_dir": "TPU-dlm",
      "clean_remote": true
    }
  ],
  "bootstrap": {
    "cwd": "TPU-dlm",
    "commands": [
      "python -V",
      "python -m pip install -U pip",
      "python -m pip install -e ."
    ],
    "fail_fast": true
  },
  "run": {
    "cwd": "TPU-dlm",
    "command": "python -c \"print('hello')\"",
    "log_path": "logs/agent_run.log"
  },
  "artifacts": [
    "logs/agent_run.log"
  ]
}
```

Then:
- tail logs via `jupyter_job_tail`
- download artifacts via `jupyter_job_collect`

## Notes

- The server binds to `127.0.0.1` by default; keep it local unless you add auth/TLS.
- `jupyter_job_*` is “Modal-like”: start a background process + tail logs (remote `logs/` by default).
