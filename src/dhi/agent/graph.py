import re
import time
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from rich.prompt import Prompt
from rich.markup import escape
from dhi.ui import console

# Import our modules
from dhi.agent.state import AgentState
from dhi.agent.llm import AI_Brain
from dhi.tools.executor import SafeExecutor
from dhi.agent.memory import MemorySystem
from dhi.agent.router import Router
from dhi.config import load_config

# --- System Prompts ---
# These go into the system role where LLMs give them the most weight.
# Local prompt is deliberately shorter to fit small context windows (2048 tokens).

LOCAL_SYSTEM_PROMPT = """You are DHI, a terminal execution engine on Arch Linux inside a Bubblewrap sandbox.

RULES:
- Output ONLY a bash code block. No explanations, no conversation.
- NEVER use sudo, pacman, yay, or apt. You have no root privileges.
- For multi-line files, use a quoted heredoc: cat << 'EOF' > filename
- For knowledge questions, use echo to print the answer.

Example — User: "list files by size"
```bash
ls -lhS
```

Example — User: "what is a tarball"
```bash
echo "A tarball is a compressed archive file created with tar, commonly using .tar.gz or .tar.bz2 extensions."
```"""

CLOUD_SYSTEM_PROMPT = """You are DHI, a strict terminal execution engine running on Arch Linux inside a Bubblewrap sandbox.

CRITICAL RULES:
1. Output ONLY executable code inside a single ```bash``` block. No prose before or after.
2. NEVER use sudo, pacman, yay, apt, or any package manager. You have NO root privileges.
3. For multi-line file creation, ALWAYS use a quoted heredoc to prevent quoting errors: cat << 'EOF' > filename
4. For knowledge/informational questions, use echo to print a concise answer.
5. Prefer simple, portable solutions. Use coreutils and standard tools when possible.

Example — User: "find duplicate files in this directory"
```bash
find . -type f -exec md5sum {} + | sort | uniq -d -w 32
```

Example — User: "explain what cron does"
```bash
echo "cron is a time-based job scheduler in Unix. You define scheduled tasks in a crontab file using the format: minute hour day month weekday command."
```"""

# Initialize Tools (Lazy Loaded)
_router = None
_executor = None
_memory = None

local_brain = None
cloud_brain = None

def get_router():
    global _router
    if _router is None: _router = Router()
    return _router

def get_executor():
    global _executor
    if _executor is None: _executor = SafeExecutor()
    return _executor

def get_memory():
    global _memory
    if _memory is None: _memory = MemorySystem()
    return _memory

_local_brain = None
_cloud_brain = None

def get_local_brain():
    global _local_brain
    if _local_brain is None:
        _local_brain = AI_Brain(mode="local")
    return _local_brain

def get_cloud_brain():
    global _cloud_brain
    if _cloud_brain is None:
        _cloud_brain = AI_Brain(mode="cloud")
    return _cloud_brain

def reload_agent():
    """Dynamically clears the AI Brain cache to force reload on next query."""
    global _local_brain, _cloud_brain
    _local_brain = None
    _cloud_brain = None

def parse_llm_output(response: str):
    """Extract executable code from LLM response.
    Only accepts properly fenced code blocks (```bash ... ```).
    Returns None if no code block is found — this triggers a retry."""
    if not response or not response.strip():
        return None
        
    code_block = re.search(r"```(?:bash|sh|python)?\n(.*?)```", response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    # No code block found. Do NOT fall back to executing raw text.
    # The LLM failed to follow the format — let the retry loop handle it.
    return None

def node_router(state: AgentState):
    # 1. Check for Exact Semantic Match (Short-circuit the LLM)
    cached_cmd = get_memory().exact_match(state['input_text'])
    if cached_cmd:
        console.print(f"[success]⚡ Semantic Cache Hit! Bypassing reasoning engine.[/success]")
        return {"plan": "executor", "command": cached_cmd, "route_confidence": 1.0}

    # 2. Run normal mathematical routing
    result = get_router().route(state['input_text'])
    decision = result["route"]
    confidence = result["confidence"]
    console.print(f"[info]ℹ Route Selected: {decision.upper()} (confidence: {confidence:.0%})[/info]")
    return {"plan": decision, "route_confidence": confidence}

def _build_user_prompt(state: AgentState, history_str: str) -> str:
    """Build the user-message prompt. User intent comes first (highest attention),
    then optional context, then error feedback at the end (recency bias)."""
    parts = [f"Task: {state['input_text']}"]

    # Add vector DB context if available
    context = get_memory().recall(state['input_text'])
    if context and context.strip():
        parts.append(f"Relevant past commands:\n{context}")

    # Add conversation history
    if history_str:
        parts.append(f"Recent history:\n{history_str}")

    # Error feedback goes last — recency bias makes the LLM focus on fixing it
    if state.get('error'):
        parts.append(f"ERROR FROM LAST ATTEMPT — fix this:\n{state['error']}")

    return "\n\n".join(parts)

def node_local_reasoner(state: AgentState):
    config = load_config()

    if config.get("stateful_local", False):
        # Truncate each message to 500 chars to prevent 4B context window collapse
        history_str = "\n".join([f"{msg.type}: {msg.content[:500] + '...' if len(msg.content) > 500 else msg.content}" for msg in state.get('messages', [])[-5:]])
    else:
        history_str = ""

    prompt = _build_user_prompt(state, history_str)
    console.print(f"[muted]⚙ Prompt Length: {len(prompt)} characters[/muted]")
    
    start_time = time.time()
    try:
        response = get_local_brain().think(prompt, system_prompt=LOCAL_SYSTEM_PROMPT)
    except Exception as e:
        return {
            "command": None,
            "command_output": f"Error connecting to Local AI: {e}",
            "error": None
        }
    end_time = time.time()
    
    console.print(f"[muted]⏱ LLM Inference Time: {end_time - start_time:.2f} seconds[/muted]")    
    command = parse_llm_output(response)
    
    return {
        "command": command, 
        "command_output": response, 
        "messages": [AIMessage(content=response)]
    }

def node_cloud_reasoner(state: AgentState):
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state.get('messages', [])[-5:]])

    prompt = _build_user_prompt(state, history_str)

    try:
        response = get_cloud_brain().think(prompt, system_prompt=CLOUD_SYSTEM_PROMPT)
    except Exception as e:
        return {
            "command": None,
            "command_output": f"Error connecting to Cloud AI: {e}\nPlease go to [s]ettings and configure your API key.",
            "error": None
        }
    command = parse_llm_output(response)
    
    return {
        "command": command, 
        "command_output": response, 
        "messages": [AIMessage(content=response)]
    }

def node_executor(state: AgentState):
    cmd = state['command']
    
    if not cmd:
        error_msg = "Error: System failed to generate an executable code block."
        console.print(f"[error]⨯ {error_msg}[/error]")
        return {"command_output": "", "error": error_msg, "retry_count": state["retry_count"] + 1}

    # Human-in-the-Loop Confirmation
    config = load_config()
    if config.get("require_confirmation", True):
        # Scan the ENTIRE command for dangerous programs — not just the start.
        # This catches: echo foo && rm -rf /, pipes, subshells, etc.
        dangerous_patterns = [
            r'\brm\b', r'\bmv\b', r'\bdd\b', r'\bmkfs\b',
            r'\bchmod\b', r'\bchown\b', r'\bkillall\b', r'\bpkill\b',
            r'\bshred\b', r'\breboot\b', r'\bshutdown\b', r'\bpoweroff\b',
        ]
        if any(re.search(pattern, cmd) for pattern in dangerous_patterns):
            console.print(f"[warning]WARNING: Destructive Command Detected[/warning]")
            console.print(f"[bold red]Command: {escape(cmd)}[/bold red]")
            proceed = Prompt.ask("Execute?", choices=["y", "n"])
            if proceed == "n":
                return {"command_output": "Execution cancelled by user.", "error": None}

    console.print(f"[info]🚀 Executing: {escape(cmd)}[/info]")
    
    # 1. Determine Network Requirement
    # Default to Zero-Trust (No Internet) unless explicitly asked
    network_keywords = ['curl', 'wget', 'download', 'git', 'http', 'https', 'api', 'ping', 'ssh']
    requires_network = any(word in state.get('input_text', '').lower() for word in network_keywords)
    
    if not requires_network:
        console.print(f"[muted]🔒 Network Sandbox Enabled (Zero-Trust)[/muted]")
        
    exec_result = get_executor().execute(cmd, requires_network=requires_network)
    
    if not exec_result["success"]:
         return {"command_output": exec_result["output"], "error": exec_result["output"], "retry_count": state["retry_count"] + 1}

    output = exec_result["output"]
    if state.get("input_text"):
        get_memory().save(state['input_text'], cmd)
    
    return {"command_output": output, "error": None}

def decide_route(state: AgentState):
    return state['plan']

def should_continue(state: AgentState):
    if state['error']:
        confidence = state.get('route_confidence', 1.0)

        if state['plan'] == 'cloud':
            if state['retry_count'] < 3:
                return "retry_cloud"
            return "end"
        else:
            # Low-confidence local routes get fewer retries before fallback offer
            local_retry_limit = 2 if confidence < 0.5 else 3

            if state['retry_count'] < local_retry_limit:
                return "retry_local"
            elif state['retry_count'] < local_retry_limit + 3:
                if state['retry_count'] == local_retry_limit:
                    console.print(f"[warning]⚠ Local Model failed {local_retry_limit} times (route confidence: {confidence:.0%}).[/warning]")
                    proceed = Prompt.ask("[bold yellow]Fallback to Cloud Model (Gemini)? (Data will be sent to the cloud)[/bold yellow]", choices=["y", "n"])
                    if proceed == "n":
                        return "end"
                return "fallback_cloud"
            return "end"
    return "end"

workflow = StateGraph(AgentState)

workflow.add_node("router", node_router)
workflow.add_node("local", node_local_reasoner)
workflow.add_node("cloud", node_cloud_reasoner)
workflow.add_node("executor", node_executor)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    decide_route,
    {"local": "local", "cloud": "cloud", "executor": "executor"}
)

workflow.add_edge("local", "executor")
workflow.add_edge("cloud", "executor")

workflow.add_conditional_edges(
    "executor",
    should_continue,
    {"retry_local": "local", "retry_cloud": "cloud", "fallback_cloud": "cloud", "end": END}
)
