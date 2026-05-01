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

def get_local_brain():
    global local_brain
    if local_brain is None:
        local_brain = AI_Brain(mode="local")
    return local_brain

def get_cloud_brain():
    global cloud_brain
    if cloud_brain is None:
        cloud_brain = AI_Brain(mode="cloud")
    return cloud_brain

def reload_agent():
    """Forces the Brains to reload their configs on the next request."""
    global local_brain, cloud_brain
    local_brain = None
    cloud_brain = None

def parse_llm_output(response: str):
    if not response or not response.strip():
        return None
        
    code_block = re.search(r"```(?:bash|sh|python)?\n(.*?)```", response, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    
    lines = response.strip().split('\n')
    if len(lines) == 1: 
         return response.strip()

    return None

def node_router(state: AgentState):
    decision = get_router().route(state['input_text'])
    console.print(f"[info]ℹ Route Selected: {decision.upper()}[/info]")
    return {"plan": decision}

def node_local_reasoner(state: AgentState):
    config = load_config()
    context = get_memory().recall(state['input_text'])

    if config.get("stateful_local", False):
        history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state.get('messages', [])[-5:]])
    else:
        history_str = "STATELESS MODE: No prior history provided to prevent local model hallucination."

    prompt = (
        f"Recent Conversation History:\n{history_str}\n\n"
        f"User Intent: {state['input_text']}\n"
        f"Context from Vector DB: {context}\n"
        f"System Context: You are Pragma-OS, a strict terminal execution engine running in an Arch Linux Bubblewrap sandbox.\n"
        f"CRITICAL RULES:\n"
        f"1. You DO NOT converse. You ONLY output executable code.\n"
        f"2. NO 'sudo', NO 'pacman', NO 'yay'. You lack root privileges.\n"
        f"3. You MUST wrap the exact bash command in a ```bash ``` block.\n"
        f"4. Do not provide explanations before or after the code block.\n"
        f"5. IMPORTANT: When writing multi-line files or scripts, ALWAYS use a quoted Heredoc to prevent quoting errors: `cat << 'EOF' > filename`.\n"
    )
    
    if state.get('error'):
        prompt += f"\n\nCRITICAL ERROR FROM LAST ATTEMPT: {state['error']}\nFix the command to resolve this error."

    console.print(f"[muted]⚙ Prompt Length: {len(prompt)} characters[/muted]")
    
    start_time = time.time()
    response = get_local_brain().think(prompt)
    end_time = time.time()
    
    console.print(f"[muted]⏱ LLM Inference Time: {end_time - start_time:.2f} seconds[/muted]")    
    command = parse_llm_output(response)
    
    return {
        "command": command, 
        "command_output": response, 
        "messages": [AIMessage(content=response)]
    }

def node_cloud_reasoner(state: AgentState):
    context = get_memory().recall(state['input_text'])
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state.get('messages', [])[-5:]])

    prompt = (
        f"Recent Conversation History:\n{history_str}\n\n"
        f"User Intent: {state['input_text']}\n"
        f"Context from Vector DB: {context}\n"
        f"System Context: You are Pragma-OS, a strict terminal execution engine running in an Arch Linux Bubblewrap sandbox.\n"
        f"CRITICAL RULES:\n"
        f"1. You DO NOT converse. You ONLY output executable code.\n"
        f"2. NO 'sudo', NO 'pacman', NO 'yay'. You lack root privileges.\n"
        f"3. You MUST wrap the exact bash command in a ```bash ``` block.\n"
        f"4. Do not provide explanations before or after the code block.\n"
        f"5. IMPORTANT: When writing multi-line files or scripts, ALWAYS use a quoted Heredoc to prevent quoting errors: `cat << 'EOF' > filename`.\n"
    )
    
    if state.get('error'):
        prompt += f"\n\nCRITICAL ERROR FROM LAST ATTEMPT: {state['error']}\nFix the code."

    response = get_cloud_brain().think(prompt)
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
        dangerous_commands = ["rm", "mv", "touch", "reboot", "shutdown"]
        if any(cmd.strip().startswith(d) for d in dangerous_commands):
            console.print(f"[warning]WARNING: Destructive Command Detected[/warning]")
            console.print(f"[bold red]Command: {escape(cmd)}[/bold red]")
            proceed = Prompt.ask("Execute?", choices=["y", "n"])
            if proceed == "n":
                return {"command_output": "Execution cancelled by user.", "error": None}

    console.print(f"[info]🚀 Executing: {escape(cmd)}[/info]")
    
    output = get_executor().execute(cmd)
    
    if "Error:" in output:
         return {"command_output": output, "error": output, "retry_count": state["retry_count"] + 1}

    if state.get("input_text"):
        get_memory().save(f"Request: {state['input_text']} -> Command: {cmd}")
    
    return {"command_output": output, "error": None}

def decide_route(state: AgentState):
    return state['plan']

def should_continue(state: AgentState):
    if state['error']:
        if state['plan'] == 'cloud':
            if state['retry_count'] < 3:
                return "retry_cloud"
            return "end"
        else:
            if state['retry_count'] < 3:
                return "retry_local"
            elif state['retry_count'] < 6:
                if state['retry_count'] == 3:
                    console.print("[warning]⚠ Local Model failed 3 times.[/warning]")
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
    {"local": "local", "cloud": "cloud"}
)

workflow.add_edge("local", "executor")
workflow.add_edge("cloud", "executor")

workflow.add_conditional_edges(
    "executor",
    should_continue,
    {"retry_local": "local", "retry_cloud": "cloud", "fallback_cloud": "cloud", "end": END}
)
