@echo off
setlocal ENABLEEXTENSIONS

REM KODO init helper (Windows)
REM - Requires DEEPGHS_ADMIN_API_KEY in the environment (or set below)
REM - Uses Versa control plane at https://versa.iseki.cloud by default

IF "%DEEPGHS_ADMIN_API_KEY%"=="" (
  echo [kodo_init] ERROR: DEEPGHS_ADMIN_API_KEY is not set.
  echo [kodo_init] Set it first, e.g.:
  echo   set DEEPGHS_ADMIN_API_KEY=YOUR_KEY
  exit /b 1
)

set VERSA_API_BASE=https://versa.iseki.cloud
set VERSA_ADMIN_KEY=%DEEPGHS_ADMIN_API_KEY%

echo [kodo_init] Listing Codex accounts...
python "%~dp0..\versa\cli.py" cloud codex-tokens --base-url "%VERSA_API_BASE%" --admin-key "%VERSA_ADMIN_KEY%"

echo [kodo_init] OK
