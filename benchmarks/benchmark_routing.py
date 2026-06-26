#!/usr/bin/env python3
"""
DHI Benchmark: Semantic Routing Accuracy
=========================================
Measures: The 96.5% number in the paper (Routing Accuracy).

Loads held-out intents from benchmark_dataset.json, runs each through
the Router, compares against ground truth, and reports accuracy with
per-tier breakdown.

This benchmark does NOT require the SLM — it only uses the embedding
model (nomic-embed-text via Ollama). Runs in ~30-60 seconds.
"""

import json
import os
import sys
import time

# Ensure we import from the local source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dhi.agent.router import Router


def load_dataset():
    dataset_path = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data["intents"]


def run_routing_benchmark():
    print("=" * 60)
    print("DHI Benchmark: Semantic Routing Accuracy")
    print("=" * 60)

    intents = load_dataset()
    router = Router()

    results = []
    latencies = []

    for entry in intents:
        intent_id = entry["id"]
        intent_text = entry["intent"]
        expected = entry["expected_route"]
        tier = entry["tier"]

        start = time.time()
        result = router.route(intent_text)
        elapsed_ms = (time.time() - start) * 1000

        predicted = result["route"]
        confidence = result["confidence"]
        correct = predicted == expected

        results.append({
            "id": intent_id,
            "intent": intent_text,
            "tier": tier,
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "correct": correct,
            "latency_ms": round(elapsed_ms, 2)
        })
        latencies.append(elapsed_ms)

        status = "✓" if correct else "✗"
        print(f"  {status} [{intent_id}] {intent_text[:50]:50s} "
              f"expected={expected:5s} got={predicted:5s} "
              f"conf={confidence:.0%} ({elapsed_ms:.0f}ms)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / total * 100

    print(f"\nOverall Accuracy: {correct_count}/{total} = {accuracy:.1f}%")

    # Per-tier breakdown
    for tier in [1, 2, 3]:
        tier_results = [r for r in results if r["tier"] == tier]
        tier_correct = sum(1 for r in tier_results if r["correct"])
        tier_total = len(tier_results)
        tier_acc = tier_correct / tier_total * 100 if tier_total > 0 else 0
        print(f"  Tier {tier}: {tier_correct}/{tier_total} = {tier_acc:.1f}%")

    # Latency stats
    import statistics
    mean_lat = statistics.mean(latencies)
    std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0
    print(f"\nRouting Latency: {mean_lat:.1f} ± {std_lat:.1f} ms (mean ± std)")
    print(f"  Min: {min(latencies):.1f} ms | Max: {max(latencies):.1f} ms")

    # Misclassified intents
    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        print(f"\nMisclassified Intents ({len(misclassified)}):")
        for r in misclassified:
            print(f"  [{r['id']}] \"{r['intent']}\"")
            print(f"         expected={r['expected']}, got={r['predicted']} "
                  f"(confidence: {r['confidence']:.0%})")

    # Confusion matrix
    print("\nConfusion Matrix:")
    tp = sum(1 for r in results if r["expected"] == "local" and r["predicted"] == "local")
    fn = sum(1 for r in results if r["expected"] == "local" and r["predicted"] == "cloud")
    fp = sum(1 for r in results if r["expected"] == "cloud" and r["predicted"] == "local")
    tn = sum(1 for r in results if r["expected"] == "cloud" and r["predicted"] == "cloud")
    print(f"                Predicted Local  Predicted Cloud")
    print(f"  Actual Local:     {tp:3d}             {fn:3d}")
    print(f"  Actual Cloud:     {fp:3d}             {tn:3d}")

    # Save detailed results to JSON
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "routing_results.json")

    output = {
        "benchmark": "routing_accuracy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_intents": total,
        "accuracy_percent": round(accuracy, 1),
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "per_tier": {},
        "detailed_results": results
    }
    for tier in [1, 2, 3]:
        tier_results = [r for r in results if r["tier"] == tier]
        tier_correct = sum(1 for r in tier_results if r["correct"])
        output["per_tier"][f"tier_{tier}"] = {
            "total": len(tier_results),
            "correct": tier_correct,
            "accuracy_percent": round(tier_correct / len(tier_results) * 100, 1) if tier_results else 0
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    run_routing_benchmark()
