#!/usr/bin/env python3
"""
DHI Benchmark: Zero-Shot and Self-Healing Execution Success
=============================================================
Measures: The 68.2% (zero-shot) and 94.1% (healed) numbers in the paper.

For each Tier 1 and Tier 2 intent in the held-out dataset, this script:
  1. Generates a command via the local SLM (gemma4:e4b-it)
  2. Parses the output for a bash code block
  3. Executes it in the Bubblewrap sandbox
  4. If it fails, feeds the error back to the SLM and retries (up to 3 times)
  5. Tracks zero-shot success, healed success, and semantic correctness

This benchmark REQUIRES:
  - Ollama running with the gemma4:e4b-it model pulled
  - Bubblewrap (bwrap) installed

Expected runtime: ~10-20 minutes depending on hardware.
"""

import json
import os
import re
import sys
import time

# Ensure we import from the local source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dhi.agent.llm import AI_Brain
from dhi.agent.graph import LOCAL_SYSTEM_PROMPT, parse_llm_output
from dhi.tools.executor import SafeExecutor

# --- Configuration ---
MODEL_NAME = "gemma4:e4b-it-q4_K_M"  # The model the paper claims to test on, matching installed tag
MAX_RETRIES = 3               # Maximum retry attempts per intent


def load_local_intents():
    """Load only Tier 1 and Tier 2 intents (local route) from the dataset."""
    dataset_path = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
    with open(dataset_path, "r") as f:
        data = json.load(f)
    # Only test intents that should be handled locally
    return [i for i in data["intents"] if i["expected_route"] == "local"]


def check_semantic_match(command, expected_patterns):
    """Check if the generated command contains any of the expected patterns."""
    if not command:
        return False
    cmd_lower = command.lower()
    return any(pattern.lower() in cmd_lower for pattern in expected_patterns)


def run_execution_benchmark():
    print("=" * 60)
    print("DHI Benchmark: Execution Success Rate")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    intents = load_local_intents()
    print(f"Loaded {len(intents)} local intents (Tier 1 + Tier 2)\n")

    # Initialize components directly — bypass the graph to avoid interactive prompts
    print("[*] Initializing SLM...")
    brain = AI_Brain(mode="local")
    # Override the model to ensure we use the paper's claimed model
    brain.llm.model = MODEL_NAME

    print("[*] Initializing Sandbox...")
    executor = SafeExecutor()

    results = []

    for idx, entry in enumerate(intents):
        intent_id = entry["id"]
        intent_text = entry["intent"]
        tier = entry["tier"]
        expected_patterns = entry["expected_command_patterns"]

        print(f"\n--- [{idx+1}/{len(intents)}] {intent_id}: \"{intent_text}\" ---")

        # --- Zero-Shot Attempt ---
        zero_shot_success = False
        healed_success = False
        semantic_match = False
        final_command = None
        retries_used = 0
        gen_latency = 0

        prompt = f"Task: {intent_text}"
        error_context = None

        for attempt in range(1 + MAX_RETRIES):  # 1 initial + MAX_RETRIES retries
            # Build prompt with error feedback if retrying
            if error_context:
                full_prompt = f"{prompt}\n\nERROR FROM LAST ATTEMPT — fix this:\n{error_context}"
            else:
                full_prompt = prompt

            # Generate command
            start = time.time()
            try:
                response = brain.think(full_prompt, system_prompt=LOCAL_SYSTEM_PROMPT)
            except Exception as e:
                print(f"  ✗ SLM Error: {e}")
                break
            gen_time = time.time() - start

            if attempt == 0:
                gen_latency = gen_time

            # Parse output
            command = parse_llm_output(response)
            if not command:
                error_context = "Failed to generate a valid bash code block. Output ONLY a ```bash``` block."
                if attempt == 0:
                    print(f"  ✗ Zero-shot: No parseable code block ({gen_time:.1f}s)")
                else:
                    print(f"  ✗ Retry {attempt}: No parseable code block")
                retries_used = attempt
                continue

            # Determine network requirement (same logic as graph.py)
            network_keywords = ['curl', 'wget', 'download', 'git', 'http', 'https', 'api', 'ping', 'ssh']
            requires_network = any(word in intent_text.lower() for word in network_keywords)

            # Execute in sandbox
            exec_result = executor.execute(command, requires_network=requires_network)

            if exec_result["success"]:
                final_command = command
                if attempt == 0:
                    zero_shot_success = True
                    print(f"  ✓ Zero-shot SUCCESS ({gen_time:.1f}s): {command[:80]}")
                else:
                    print(f"  ✓ Healed on retry {attempt}: {command[:80]}")
                healed_success = True
                semantic_match = check_semantic_match(command, expected_patterns)
                retries_used = attempt
                break
            else:
                error_context = exec_result["output"]
                if attempt == 0:
                    print(f"  ✗ Zero-shot FAILED ({gen_time:.1f}s): {command[:60]}")
                    print(f"    Error: {error_context[:100]}")
                else:
                    print(f"  ✗ Retry {attempt} FAILED: {command[:60]}")
                retries_used = attempt

        results.append({
            "id": intent_id,
            "intent": intent_text,
            "tier": tier,
            "zero_shot_success": zero_shot_success,
            "healed_success": healed_success,
            "semantic_match": semantic_match,
            "retries_used": retries_used,
            "final_command": final_command,
            "gen_latency_s": round(gen_latency, 2)
        })

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    total = len(results)

    zero_shot_count = sum(1 for r in results if r["zero_shot_success"])
    healed_count = sum(1 for r in results if r["healed_success"])
    semantic_count = sum(1 for r in results if r["semantic_match"])

    zero_shot_rate = zero_shot_count / total * 100
    healed_rate = healed_count / total * 100
    semantic_rate = semantic_count / total * 100

    print(f"\nTotal Intents Tested: {total}")
    print(f"  Zero-Shot Success (exit 0, first try):  {zero_shot_count}/{total} = {zero_shot_rate:.1f}%")
    print(f"  Healed Success (exit 0, ≤3 retries):    {healed_count}/{total} = {healed_rate:.1f}%")
    print(f"  Semantic Success (exit 0 + pattern):     {semantic_count}/{total} = {semantic_rate:.1f}%")

    # Per-tier breakdown
    for tier in [1, 2]:
        tier_results = [r for r in results if r["tier"] == tier]
        if not tier_results:
            continue
        t_total = len(tier_results)
        t_zs = sum(1 for r in tier_results if r["zero_shot_success"])
        t_hs = sum(1 for r in tier_results if r["healed_success"])
        t_sm = sum(1 for r in tier_results if r["semantic_match"])
        print(f"\n  Tier {tier} ({t_total} intents):")
        print(f"    Zero-Shot:  {t_zs}/{t_total} = {t_zs/t_total*100:.1f}%")
        print(f"    Healed:     {t_hs}/{t_total} = {t_hs/t_total*100:.1f}%")
        print(f"    Semantic:   {t_sm}/{t_total} = {t_sm/t_total*100:.1f}%")

    # Retry distribution
    retry_counts = [r["retries_used"] for r in results if r["healed_success"]]
    if retry_counts:
        import statistics
        avg_retries = statistics.mean(retry_counts)
        print(f"\n  Avg retries needed (for healed successes): {avg_retries:.1f}")

    # Generation latency
    latencies = [r["gen_latency_s"] for r in results if r["gen_latency_s"] > 0]
    if latencies:
        import statistics
        mean_lat = statistics.mean(latencies)
        std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0
        print(f"\n  Generation Latency: {mean_lat:.1f} ± {std_lat:.1f} s (mean ± std)")

    # Failed intents
    failed = [r for r in results if not r["healed_success"]]
    if failed:
        print(f"\nFailed Intents ({len(failed)}):")
        for r in failed:
            print(f"  [{r['id']}] \"{r['intent']}\"")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "execution_results.json")

    output = {
        "benchmark": "execution_success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL_NAME,
        "max_retries": MAX_RETRIES,
        "total_intents": total,
        "zero_shot_rate_percent": round(zero_shot_rate, 1),
        "healed_rate_percent": round(healed_rate, 1),
        "semantic_rate_percent": round(semantic_rate, 1),
        "detailed_results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    run_execution_benchmark()
