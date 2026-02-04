@echo off
setlocal ENABLEEXTENSIONS

REM Watches %USERPROFILE%\.codex\auth.json and syncs it into Versa KODO SSOT.

IF "%DEEPGHS_ADMIN_API_KEY%"=="" (
  echo [kodo_watch] ERROR: DEEPGHS_ADMIN_API_KEY is not set.
  exit /b 1
)

set VERSA_API_BASE=https://versa.iseki.cloud
set VERSA_ADMIN_KEY=%DEEPGHS_ADMIN_API_KEY%

python "%~dp0kodo_watch_auth.py"

