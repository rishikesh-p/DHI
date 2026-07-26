import re
import time
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from rich.prompt import Prompt
from rich.markup import escape
from dhi.ui import console

# Import our modules
from dhi.agent.state import AgentState
from dhi.config import load_config

# System Prompts
# Define behavior for the LLMs.
# The local prompt is kept short for small context windows (2048 tokens).

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

# Lazy-loaded tools
_router = None
_executor = None
_memory = None
_cached_config = None

local_brain = None
cloud_brain = None

def get_config():
    """Return the cached config, loading from disk on first call."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config

def get_router():
    global _router
    if _router is None:
        from dhi.agent.router import Router
        _router = Router()
    return _router

def get_executor():
    global _executor
    if _executor is None:
        from dhi.tools.executor import SafeExecutor
        _executor = SafeExecutor()
    return _executor

def get_memory():
    global _memory
    if _memory is None:
        from dhi.agent.memory import MemorySystem
        _memory = MemorySystem()
    return _memory

_local_brain = None
_cloud_brain = None

def get_local_brain():
    global _local_brain
    if _local_brain is None:
        from dhi.agent.llm import AI_Brain
        _local_brain = AI_Brain(mode="local")
    return _local_brain

def get_cloud_brain():
    global _cloud_brain
    if _cloud_brain is None:
        from dhi.agent.llm import AI_Brain
        _cloud_brain = AI_Brain(mode="cloud")
    return _cloud_brain

def reload_agent():
    """Clear the AI Brain and config caches to force reload on next query."""
    global _local_brain, _cloud_brain, _cached_config
    _local_brain = None
    _cloud_brain = None
    _cached_config = None

def parse_llm_output(response: str):
    """Extract executable code from LLM response.
    Only accept properly fenced code blocks (```bash ... ```).
    Return None if no code block is found to trigger a retry."""
    if not response or not response.strip():
        return None
        
    code_block = re.search(r"```[a-zA-Z]*[ \t]*\n(.*?)```", response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    # Do not execute raw text if no code block is found.
    # Allow the retry loop to handle the format failure.
    return None

def node_router(state: AgentState):
    # Compute vector ONCE for the entire graph turn
    input_vector = get_router().embedder.embed_query(state['input_text'])

    # Check for exact semantic match to bypass LLM
    cached_cmd = get_memory().exact_match_vec(input_vector)
    if cached_cmd:
        console.print(f"[success]⚡ Semantic Cache Hit! Bypassing reasoning engine.[/success]")
        return {"plan": "executor", "command": cached_cmd, "route_confidence": 1.0, "input_vector": input_vector}

    # Run mathematical routing
    result = get_router().route_vec(input_vector)
    decision = result["route"]
    confidence = result["confidence"]
    console.print(f"[info]ℹ Route Selected: {decision.upper()} (confidence: {confidence:.0%})[/info]")
    return {"plan": decision, "route_confidence": confidence, "input_vector": input_vector}

def _build_user_prompt(state: AgentState, history_str: str) -> str:
    """Build the user-message prompt.
    Order components by attention priority: user intent, context, and error feedback."""
    parts = [f"Task: {state['input_text']}"]

    # Add vector DB context if available
    input_vector = state.get('input_vector')
    if input_vector:
        context = get_memory().recall_vec(input_vector)
    else:
        context = get_memory().recall(state['input_text'])
    if context and context.strip():
        parts.append(f"Relevant past commands:\n{context}")

    # Add conversation history
    if history_str:
        parts.append(f"Recent history:\n{history_str}")

    # Append error feedback last to leverage recency bias.
    if state.get('error'):
        parts.append(f"ERROR FROM LAST ATTEMPT — fix this:\n{state['error']}")

    return "\n\n".join(parts)

def node_local_reasoner(state: AgentState):
    config = get_config()

    if config.get("stateful_local", False):
        # Truncate each message to 500 chars to prevent small context window collapse.
        history_str = "\n".join([f"{msg.type}: {msg.content[:500] + '...' if len(msg.content) > 500 else msg.content}" for msg in state.get('messages', [])[-5:]])
    else:
        history_str = ""

    prompt = _build_user_prompt(state, history_str)
    console.print(f"[muted]⚙ Prompt Length: {len(prompt)} characters[/muted]")
    
    start_time = time.time()
    try:
        response = get_local_brain().think(prompt, system_prompt=LOCAL_SYSTEM_PROMPT)
    except Exception as e:
        error_msg = f"Error connecting to Local AI: {e}"
        return {
            "command": None,
            "command_output": error_msg,
            "error": error_msg
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
        error_str = str(e)
        if "API_KEY_MISSING" in error_str or "API key not valid" in error_str or "API key required" in error_str or "401" in error_str or "auth" in error_str.lower():
            console.print(f"\n[warning]⚠ Cloud API Key Missing or Invalid[/warning]")
            
            from rich.prompt import Prompt
            from dhi.config import load_config, save_config
            config = load_config()
            
            console.print("[info]Providers: [1] Google Gemini, [2] Anthropic Claude, [3] OpenAI (or compatible)[/info]")
            provider_choice = Prompt.ask("Select Provider", choices=["1", "2", "3"], default="1")
            provider_map = {"1": "google", "2": "anthropic", "3": "openai"}
            selected_provider = provider_map[provider_choice]
            
            help_links = {
                "google": "Get your key here: https://aistudio.google.com/app/apikey",
                "anthropic": "Get your key here: https://console.anthropic.com/settings/keys",
                "openai": "Get your key here: https://platform.openai.com/api-keys"
            }
            console.print(f"[dim]{help_links[selected_provider]}[/dim]")
            
            defaults = {"google": "gemini-2.5-flash", "anthropic": "claude-3-5-sonnet-20240620", "openai": "gpt-4o-mini"}
            model = Prompt.ask(f"Model Name", default=defaults[selected_provider])
            
            base_url = ""
            if selected_provider == "openai":
                base_url = Prompt.ask(f"Custom Base URL (Leave empty for default OpenAI)")
                
            api_key = Prompt.ask(f"Enter {selected_provider.capitalize()} API Key (or press Enter to cancel)", password=True)
            
            if api_key.strip():
                config["cloud_provider"] = selected_provider
                config["cloud_model"] = model.strip()
                config["cloud_base_url"] = base_url.strip()
                config["cloud_api_key"] = api_key.strip()
                save_config(config)
                
                # Force reload of the cloud brain so it picks up the new config
                global _cloud_brain, _cached_config
                _cloud_brain = None
                _cached_config = None
                
                console.print("[success]✓ Configuration saved! Retrying...[/success]\n")
                try:
                    response = get_cloud_brain().think(prompt, system_prompt=CLOUD_SYSTEM_PROMPT)
                except Exception as retry_e:
                    error_msg = f"Error connecting to Cloud AI: {retry_e}\nPlease check your API key in [s]ettings."
                    return {"command": None, "command_output": error_msg, "error": error_msg}
            else:
                error_msg = "Cloud AI cancelled: Missing API Key."
                return {"command": None, "command_output": error_msg, "error": error_msg}
        else:
            error_msg = f"Error connecting to Cloud AI: {e}\nPlease go to [s]ettings and configure your API key."
            return {
                "command": None,
                "command_output": error_msg,
                "error": error_msg
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
        if state.get('error'):
            # A reasoner node already reported a critical failure (like a connection error).
            console.print(f"[error]⨯ {state['error']}[/error]")
            return {"retry_count": state["retry_count"] + 1}
            
        error_msg = "Error: System failed to generate an executable code block."
        console.print(f"[error]⨯ {error_msg}[/error]")
        return {"command_output": "", "error": error_msg, "retry_count": state["retry_count"] + 1}

    # Request human-in-the-loop confirmation.
    config = get_config()
    if config.get("require_confirmation", True):
        # Scan the entire command for dangerous programs (e.g., in pipes or subshells).
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
    
    # Determine network requirement.
    # Default to Zero-Trust (no Internet) unless explicitly requested.
    network_keywords = [
        'curl', 'wget', 'download', 'git', 'http', 'https', 'api', 'ping', 'ssh',
        'pip', 'npm', 'cargo', 'docker', 'install', 'update', 'upgrade', 'clone', 'fetch', 'pull', 'push'
    ]
    search_text = (state.get('input_text', '') + ' ' + cmd).lower()
    requires_network = state.get('force_network', False) or any(word in search_text for word in network_keywords)
    
    if not requires_network:
        console.print(f"[muted]🔒 Network Sandbox Enabled (Zero-Trust)[/muted]")
        
    exec_result = get_executor().execute(cmd, requires_network=requires_network)
    
    if not exec_result["success"]:
        output = exec_result["output"]
        
        # Reactive Sandboxing Check
        if not requires_network:
            net_errors = ["Network is unreachable", "Name or service not known", "Temporary failure in name resolution", "Could not resolve host", "Connection refused"]
            if any(err in output for err in net_errors):
                console.print("[warning]⚠ Network access was denied by the Zero-Trust Sandbox. Retrying with Network Enabled...[/warning]")
                return {"force_network": True, "error": None}
                
        return {"command_output": output, "error": output, "retry_count": state["retry_count"] + 1}

    output = exec_result["output"]
    if state.get("input_text"):
        input_vector = state.get("input_vector")
        if input_vector:
            get_memory().save_vec(state['input_text'], input_vector, cmd)
        else:
            get_memory().save(state['input_text'], cmd)
    
    return {"command_output": output, "error": None}

def decide_route(state: AgentState):
    return state['plan']

def should_continue(state: AgentState):
    if state.get('force_network') and not state.get('error'):
        return "retry_execution"
        
    if state.get('error'):
        # Connection/API errors are not recoverable by retrying.
        if "Error connecting to" in state['error']:
            return "end"

        confidence = state.get('route_confidence', 1.0)

        if state['plan'] == 'cloud':
            if state['retry_count'] < 3:
                return "retry_cloud"
            return "end"
        else:
            # Offer fallback sooner for low-confidence local routes.
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
    {"retry_local": "local", "retry_cloud": "cloud", "fallback_cloud": "cloud", "retry_execution": "executor", "end": END}
)
