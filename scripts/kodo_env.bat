@echo off
setlocal ENABLEEXTENSIONS

REM Sets environment variables so OpenAI SDKs can use Versa KODO reverse-proxy.
REM After running this in a terminal, run your normal OpenAI client code.

IF "%DEEPGHS_ADMIN_API_KEY%"=="" (
  echo [kodo_env] ERROR: DEEPGHS_ADMIN_API_KEY is not set.
  echo [kodo_env] Example:
  echo   set DEEPGHS_ADMIN_API_KEY=YOUR_KEY
  exit /b 1
)

set OPENAI_BASE_URL=https://versa.iseki.cloud/kodo/v1
set OPENAI_API_KEY=%DEEPGHS_ADMIN_API_KEY%

echo [kodo_env] OPENAI_BASE_URL=%OPENAI_BASE_URL%
echo [kodo_env] OPENAI_API_KEY is set from DEEPGHS_ADMIN_API_KEY
echo [kodo_env] Now run your OpenAI client in this same terminal.

