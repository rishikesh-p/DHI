#!/usr/bin/env python3
"""
DHI Benchmark: LanceDB Semantic Cache Latency
===============================================
Measures: The 31.84ms number in the paper (Cache Hit Latency).

Seeds a temporary LanceDB instance with known intent→command pairs,
then measures lookup latency across multiple trials. Reports mean,
std, and p95 latency.

This benchmark only uses the embedding model (nomic-embed-text).
Runs in ~30 seconds.
"""

import json
import os
import sys
import time
import tempfile
import shutil

# Ensure we import from the local source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dhi.agent.memory import MemorySystem

# --- Seed Data ---
# 20 realistic intent→command pairs to populate the cache
SEED_ENTRIES = [
    "Request: list files -> Command: ls -la",
    "Request: show disk usage -> Command: df -h",
    "Request: show running processes -> Command: ps aux",
    "Request: what is the date -> Command: date",
    "Request: show memory usage -> Command: free -h",
    "Request: count lines in file -> Command: wc -l file.txt",
    "Request: find large files -> Command: find . -size +100M",
    "Request: show system uptime -> Command: uptime",
    "Request: display current user -> Command: whoami",
    "Request: check kernel version -> Command: uname -r",
    "Request: show current directory -> Command: pwd",
    "Request: show environment variables -> Command: env",
    "Request: display file contents -> Command: cat file.txt",
    "Request: search for text in files -> Command: grep -r 'text' .",
    "Request: show network interfaces -> Command: ip addr",
    "Request: kill a process -> Command: kill -9 1234",
    "Request: create a directory -> Command: mkdir new_folder",
    "Request: rename a file -> Command: mv old.txt new.txt",
    "Request: show last 10 lines of log -> Command: tail -10 app.log",
    "Request: sort file contents -> Command: sort data.txt",
]

# Queries to test — mix of exact matches, near matches, and misses
TEST_QUERIES = [
    # Near-exact matches (should hit cache with distance < 0.05)
    {"query": "list files", "expect_hit": True},
    {"query": "show disk usage", "expect_hit": True},
    {"query": "show running processes", "expect_hit": True},
    {"query": "what is today's date", "expect_hit": True},
    {"query": "show memory usage", "expect_hit": True},
    {"query": "count lines in a file", "expect_hit": True},
    {"query": "show system uptime", "expect_hit": True},
    {"query": "display current user", "expect_hit": True},
    {"query": "check the kernel version", "expect_hit": True},
    {"query": "show current directory", "expect_hit": True},

    # Near matches (may or may not hit depending on threshold)
    {"query": "list all my files with details", "expect_hit": False},
    {"query": "how much free disk space do I have", "expect_hit": False},
    {"query": "show me the processes using the most CPU", "expect_hit": False},
    {"query": "display file contents of readme", "expect_hit": False},
    {"query": "search for the word error in log files", "expect_hit": False},

    # Clear misses (should not hit cache)
    {"query": "write a python web scraper", "expect_hit": False},
    {"query": "build a docker container", "expect_hit": False},
    {"query": "create an HTML dashboard", "expect_hit": False},
    {"query": "set up a cron job for backups", "expect_hit": False},
    {"query": "generate a complex regex pattern", "expect_hit": False},
]


def run_cache_benchmark():
    print("=" * 60)
    print("DHI Benchmark: LanceDB Semantic Cache Latency")
    print("=" * 60)

    # Use a temporary directory so we don't pollute the real DB
    tmp_dir = tempfile.mkdtemp(prefix="dhi_bench_cache_")
    print(f"[*] Using temporary DB at: {tmp_dir}")

    try:
        # Initialize with temporary path
        memory = MemorySystem(db_path=tmp_dir)

        # Seed the database
        print(f"[*] Seeding {len(SEED_ENTRIES)} entries...")
        for entry in SEED_ENTRIES:
            memory.save(entry)
        print("[*] Seeding complete.\n")

        # --- Benchmark: exact_match latency ---
        print("[1] Measuring exact_match() latency (cache hit path)...")
        hit_latencies = []
        miss_latencies = []
        hits = 0
        misses = 0

        # Run multiple passes for statistical significance
        NUM_PASSES = 3
        for pass_num in range(NUM_PASSES):
            for test in TEST_QUERIES:
                query = test["query"]

                start = time.time()
                result = memory.exact_match(query)
                elapsed_ms = (time.time() - start) * 1000

                is_hit = result is not None
                if is_hit:
                    hit_latencies.append(elapsed_ms)
                    hits += 1
                else:
                    miss_latencies.append(elapsed_ms)
                    misses += 1

                if pass_num == 0:  # Only print on first pass
                    status = "HIT " if is_hit else "MISS"
                    cmd_preview = result[:50] if result else "—"
                    print(f"  [{status}] \"{query[:45]:45s}\" → {cmd_preview} ({elapsed_ms:.1f}ms)")

        # --- Benchmark: recall latency ---
        print(f"\n[2] Measuring recall() latency (RAG retrieval path)...")
        recall_latencies = []
        for pass_num in range(NUM_PASSES):
            for test in TEST_QUERIES:
                start = time.time()
                memory.recall(test["query"])
                elapsed_ms = (time.time() - start) * 1000
                recall_latencies.append(elapsed_ms)

        # --- Summary ---
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        import statistics

        all_latencies = hit_latencies + miss_latencies

        if all_latencies:
            mean_all = statistics.mean(all_latencies)
            std_all = statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0
            sorted_all = sorted(all_latencies)
            p95_idx = int(len(sorted_all) * 0.95)
            p95_all = sorted_all[min(p95_idx, len(sorted_all) - 1)]
            print(f"\nexact_match() Latency (all queries, {NUM_PASSES} passes):")
            print(f"  Mean: {mean_all:.2f} ms | Std: {std_all:.2f} ms | P95: {p95_all:.2f} ms")
            print(f"  Min: {min(all_latencies):.2f} ms | Max: {max(all_latencies):.2f} ms")

        if hit_latencies:
            mean_hit = statistics.mean(hit_latencies)
            std_hit = statistics.stdev(hit_latencies) if len(hit_latencies) > 1 else 0
            print(f"\n  Cache HITs only ({len(hit_latencies)} lookups):")
            print(f"    Mean: {mean_hit:.2f} ms | Std: {std_hit:.2f} ms")

        if miss_latencies:
            mean_miss = statistics.mean(miss_latencies)
            std_miss = statistics.stdev(miss_latencies) if len(miss_latencies) > 1 else 0
            print(f"\n  Cache MISSes only ({len(miss_latencies)} lookups):")
            print(f"    Mean: {mean_miss:.2f} ms | Std: {std_miss:.2f} ms")

        total_lookups = hits + misses
        print(f"\n  Hit Rate: {hits}/{total_lookups} = {hits/total_lookups*100:.1f}%")

        if recall_latencies:
            mean_recall = statistics.mean(recall_latencies)
            std_recall = statistics.stdev(recall_latencies) if len(recall_latencies) > 1 else 0
            print(f"\nrecall() Latency (RAG retrieval, {NUM_PASSES} passes):")
            print(f"  Mean: {mean_recall:.2f} ms | Std: {std_recall:.2f} ms")

        # Save results
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, "cache_results.json")

        output = {
            "benchmark": "cache_latency",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_seed_entries": len(SEED_ENTRIES),
            "num_test_queries": len(TEST_QUERIES),
            "num_passes": NUM_PASSES,
            "exact_match_mean_ms": round(mean_all, 2) if all_latencies else None,
            "exact_match_std_ms": round(std_all, 2) if all_latencies else None,
            "exact_match_p95_ms": round(p95_all, 2) if all_latencies else None,
            "hit_mean_ms": round(mean_hit, 2) if hit_latencies else None,
            "miss_mean_ms": round(mean_miss, 2) if miss_latencies else None,
            "recall_mean_ms": round(mean_recall, 2) if recall_latencies else None,
            "recall_std_ms": round(std_recall, 2) if recall_latencies else None,
            "hit_rate_percent": round(hits / total_lookups * 100, 1)
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")

    finally:
        # Clean up temporary DB
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n[*] Cleaned up temporary DB.")


if __name__ == "__main__":
    run_cache_benchmark()
