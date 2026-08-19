from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from leanproof.config import build_model_registry, load_config
from leanproof.models import OpenAICompatibleProofModel

MODEL_ALIAS = "deepseek_r"

# Replace this with the EXACT theorem that previously hit the 300 s timeout.
FIXED_THEOREM = r"""
theorem lean_workbook_plus_2102 (θ : ℝ) : sin (2 * θ) = 2 * tan θ / (1 + tan θ ^ 2)
"""


def main() -> int:
    runtime_config = load_config(PROJECT_ROOT / "configs" / "default.toml")

    registry = build_model_registry(
        runtime_config,
        dotenv_path=PROJECT_ROOT / ".env",
        required_alias=MODEL_ALIAS,
    )
    config = registry.get(MODEL_ALIAS)

    # Diagnostic client:
    # - no automatic SDK retries
    # - no client-side timeout
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        max_retries=0,
        timeout=None,
    )

    # Reuse the project's exact generation prompt and response parsing.
    model = OpenAICompatibleProofModel(
        config,
        client=client,
    )

    print(f"Model: {config.model}")
    print(f"Endpoint: {config.base_url}")
    print(f"reasoning_split: {config.reasoning_split}")
    print("Client timeout: disabled")
    print()
    print("Theorem:")
    print(FIXED_THEOREM.strip())
    print()
    print("Sending request...")
    print("Press Ctrl+C to abort manually.")
    print()

    started = time.perf_counter()

    try:
        result = model.generate_proof(FIXED_THEOREM.strip())
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - started
        print()
        print(f"Interrupted after {elapsed:.2f} s ({elapsed / 60:.2f} min)")
        return 130
    except Exception as error: # noqa: BLE001
        elapsed = time.perf_counter() - started
        print()
        print(f"Request failed after {elapsed:.2f} s")
        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error!r}")

        cause = error.__cause__ or error.__context__
        if cause is not None:
            print(f"Cause type: {type(cause).__name__}")
            print(f"Cause: {cause!r}")

        return 1

    elapsed = time.perf_counter() - started

    print("=" * 72)
    print("COMPLETED")
    print("=" * 72)
    print(f"Wall-clock time:     {elapsed:.2f} s")
    print(f"Wall-clock minutes:  {elapsed / 60:.2f} min")
    print(f"Model latency_ms:    {result.latency_ms}")
    print(f"Prompt tokens:       {result.prompt_tokens}")
    print(f"Completion tokens:   {result.completion_tokens}")

    reasoning = result.native_reasoning_output or ""
    plan_output = result.plan_output or ""

    print(f"Reasoning chars:     {len(reasoning)}")
    print(f"Plan chars:          {len(plan_output)}")
    print(f"Proof chars:         {len(result.proof_output)}")

    print()
    print("=" * 72)
    print("REASONING")
    print("=" * 72)
    print(reasoning if reasoning else "<none>")

    print()
    print("=" * 72)
    print("PLAN")
    print("=" * 72)
    print(plan_output if plan_output else "<none>")

    print()
    print("=" * 72)
    print("PROOF OUTPUT")
    print("=" * 72)
    print(result.proof_output)

    output_directory = PROJECT_ROOT / "artifacts" / "diagnostics"
    output_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = output_directory / f"deepseek_latency_{timestamp}.json"

    record = {
        "model_alias": MODEL_ALIAS,
        "model": config.model,
        "theorem": FIXED_THEOREM.strip(),
        "wall_clock_seconds": elapsed,
        "model_latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_output": result.native_reasoning_output,
        "plan_output": result.plan_output,
        "proof_output": result.proof_output,
        "raw_output": result.raw_output,
    }

    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Saved result: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
