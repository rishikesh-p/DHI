import time
import sys
import os

# Hardcode the path to ensure it uses your bleeding-edge code
sys.path.insert(0, "/home/rishikesh/linux-ai/src")

from dhi.agent.router import Router
from dhi.agent.memory import MemorySystem
from dhi.agent.llm import AI_Brain
from dhi.agent.graph import LOCAL_SYSTEM_PROMPT

print("=== DHI Empirical Benchmark Suite ===")
print("Gathering data for IEEE Paper Section 5 (Results)\n")

# 1. Routing Latency
print("[1] Measuring Semantic Routing Latency...")
router = Router()
start = time.time()
result = router.route("Find all PDFs modified yesterday and compress them")
end = time.time()
routing_ms = (end - start) * 1000

route_str = result['route'] if isinstance(result, dict) else result
print(f"    -> Route Selected: {route_str}")
print(f"    -> Routing Time: {routing_ms:.2f} ms\n")

# 2. Vector Cache (LanceDB) Latency
print("[2] Measuring LanceDB Semantic Cache Latency...")
memory = MemorySystem()
memory.save("Request: list files -> Command: ls -la")

start = time.time()
cached_cmd = memory.exact_match("list files")
end = time.time()
cache_ms = (end - start) * 1000
print(f"    -> Cache Hit: {cached_cmd is not None}")
print(f"    -> Cache Lookup Time: {cache_ms:.2f} ms\n")

# 3. Local Inference Latency
print("[3] Measuring Local SLM Generation Latency (Qwen-4B / Gemma4)...")
local_llm = AI_Brain(mode="local")
start = time.time()
try:
    # We now pass the STRICT system prompt from graph.py to prevent conversational hallucination!
    response = local_llm.think("list files in bash", system_prompt=LOCAL_SYSTEM_PROMPT)
    end = time.time()
    inference_ms = (end - start) * 1000
    print(f"    -> Inference Time: {inference_ms:.2f} ms ({inference_ms/1000:.2f} seconds)")
    print(f"    -> Model Output: \n{response}")
except Exception as e:
    print(f"    -> Failed to run local inference: {e}")

print("\n=============================================")
print("BENCHMARK COMPLETE.")
