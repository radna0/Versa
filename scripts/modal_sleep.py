import os
import time

import modal

app = modal.App("versa-validate-sleep")


@app.function()
def sleeper(seconds: int) -> int:
    print(f"sleep_seconds={seconds}")
    time.sleep(seconds)
    return seconds


@app.local_entrypoint()
def main() -> None:
    seconds_s = (os.getenv("VERSA_SLEEP_SECONDS") or "600").strip()
    try:
        seconds = int(seconds_s)
    except Exception:
        seconds = 600
    print(f"sleeping {seconds}s")
    print("sleep result", sleeper.remote(seconds))

