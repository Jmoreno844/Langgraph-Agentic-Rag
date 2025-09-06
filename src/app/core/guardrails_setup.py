import asyncio
import sys
import subprocess
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def ensure_guardrails_installed() -> bool:
    """Ensure Guardrails Hub validators are installed (idempotent)."""
    try:
        # Check if validators are already available
        from guardrails.hub import RestrictToTopic, DetectPII

        return True
    except ImportError:
        pass

    try:
        # Install validators
        validators = [
            [
                "guardrails",
                "hub",
                "install",
                "hub://tryolabs/restricttotopic",
                "--quiet",
            ],
            ["guardrails", "hub", "install", "hub://guardrails/detect_pii", "--quiet"],
        ]

        for validator in validators:
            result = subprocess.run(
                validator, capture_output=True, text=True, check=True
            )

        # Clear import cache and re-import
        import importlib

        if "guardrails.hub" in sys.modules:
            importlib.reload(sys.modules["guardrails.hub"])

        return True
    except Exception as e:
        print(f"Failed to install Guardrails validators: {e}")
        return False


async def initialize_guardrails() -> None:
    """Async wrapper for Guardrails initialization."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ensure_guardrails_installed)
