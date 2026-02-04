# Versa CLI

`versa` is a Modal-like CLI that can run jobs on providers (Modal, Kaggle, ...) and can
optionally talk to a Versa control plane (`https://versa.iseki.cloud`) for account
selection, quotas, and tracking.

## Install

Editable install (recommended for development):

```powershell
pip install -e .
```

Install from Git:

```powershell
pip install "git+https://github.com/radna0/Versa.git"
```

Both install methods add the `versa` command.

Windows note: if `versa` is not found, either add your user Scripts directory to `PATH`
(typically `%APPDATA%\\Python\\Python3xx\\Scripts`), or run via module:

```powershell
python -m versa --help
```

## Core Idea (How Agents Use Versa)

Versa is a *control plane* (`versa.iseki.cloud`) + a *runner CLI* (`versa`).

When you run a job via `versa cloud run ...`:

1) CLI asks Versa Worker to create a run (`/api/runs`).
2) Versa Worker *selects an account/profile automatically* (unless you pin one).
3) CLI fetches short-lived credentials for that assigned account/profile.
4) CLI executes the provider tool locally (e.g. `python -m modal run ...`) and streams logs back to Versa.
5) You can cancel via `versa cloud kill <vr_...>`; the runner will stop both:
   - the local process, and
   - the remote Modal app (`modal app stop ap-...`) once the run URL is known.

**Important:** for cancellation to work, the runner process must still be alive to observe the kill flag.

## Usage (Cloud)

Environment:

- `VERSA_API_BASE` (default: `https://versa.iseki.cloud`)
- `VERSA_ADMIN_KEY` (required for admin endpoints)

Or set them once via:

```powershell
versa login --prompt
```

Examples:

- `versa cloud usage`
- `versa cloud runs --limit 20`
- `versa cloud run modal my_app.py::main -- --arg1 foo`
- `versa cloud logs <vr_...>`
- `versa cloud kill <vr_...> --wait`

### Structured Cloud Subcommands (Recommended)

You can use either the legacy flat commands (`modal-import`, `kaggle-tokens`, ...) **or** the structured provider groups:

- `versa cloud modal ...`
- `versa cloud kaggle ...`
- `versa cloud codex ...`
- `versa cloud kodo ...`

Examples:

- `versa cloud modal tokens`
- `versa cloud modal summary`
- `versa cloud modal run scripts/modal_square.py::main`
- `versa cloud modal gpus`

## Modal Quickstart (No Manual Profile Switching)

### 1) Import Modal profiles (admin-only, one-time)

Import a single profile:

```powershell
versa cloud modal-import phamtest2 <token_id> <token_secret> --budget-usd 100
```

Or bulk import from TSV:

```powershell
versa cloud modal-import-b3 ~/.versa/modal_profiles_import.txt
```

### 2) (Optional) Probe billing / validity

This updates `valid` / `has_credit` signals for schedulers + UI:

```powershell
versa cloud modal-probe --profiles-file ~/.versa/modal_profiles_import.txt --concurrency 10
```

### 3) Launch a job (Versa auto-picks a profile)

No `--profile` means Versa selects an eligible profile automatically:

```powershell
versa cloud run modal scripts/modal_square.py::main
```

### 4) Pin a specific profile (optional)

```powershell
versa cloud run-modal scripts/modal_square.py::main --profile phamtest2
```

### 5) GPU jobs

You can request a GPU from Versa:

```powershell
versa cloud run-modal scripts/modal_gpu_smoke.py::main --gpu-type L4 --gpu-count 1
```

List GPU types supported by your local Modal SDK:

```powershell
versa modal gpus
versa cloud modal gpus
```

Note: Modal GPU provisioning is defined in Python code. Versa passes `VERSA_MODAL_GPU_TYPE` /
`VERSA_MODAL_GPU_COUNT` env vars; your script should read those and set `@app.function(gpu=...)`.

### 6) List / logs / kill

List runs:

```powershell
versa cloud runs --provider modal --limit 20
versa cloud runs --provider modal --profile phamtest2 --limit 20
```

Tail logs:

```powershell
versa cloud logs <vr_...> --follow
```

Kill:

```powershell
versa cloud kill <vr_...> --wait
```

## Validation / Regression Tests

Validate many profiles end-to-end:

```powershell
versa cloud modal-validate --only-valid --concurrency 4 --summary-only
versa cloud modal-validate --gpu --concurrency 3 --summary-only
```

Token inventory:

- `versa cloud kaggle-summary`
- `versa cloud kaggle-tokens`
- `versa cloud kaggle-import <username> <kaggle.json> --max-concurrency 5`
- `versa cloud modal-summary`
- `versa cloud modal-tokens`
- `versa cloud modal-import <profile> <token_id> <token_secret>`

## Modal Credits Remaining (No Cookies)

Versa does **not** require (or store) your `modal-session` cookie.

Instead:

- You optionally provide a per-profile `budget_usd` (e.g. `100`) when importing profiles (or via a TSV file).
- Versa-CLI uses the **Modal Python SDK billing report** (month-to-date) when available and computes:
  - `mtd_cost_usd`
  - `budget_remaining_usd = budget_usd - mtd_cost_usd`
  - `has_credit = budget_remaining_usd > 0`

Create `~/.versa/modal_profiles_import.txt` (TSV, one per line):

`profile<TAB>token_id<TAB>token_secret<TAB>budget_usd`

Then run:

- `versa cloud modal-probe --profiles-file ~/.versa/modal_profiles_import.txt --concurrency 10`

Optional: web-accurate "spend limit" + "cycle credit"

- `versa cloud modal-probe-web --profiles-file ~/.versa/modal_profiles_import.txt --modal-session "$env:MODAL_SESSION"`

Note: Versa does **not** store the cookie; it only stores derived numeric fields in `quota_json`.

