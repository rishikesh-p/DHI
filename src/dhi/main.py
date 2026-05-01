import sys
import os
import sqlite3
import subprocess
from rich.prompt import Prompt
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.align import Align
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from dhi.ui import console
from dhi.audio.ears import Ear
from dhi.agent.graph import workflow, reload_agent
from dhi.config import load_config, save_config

def startup_checks():
    console.print("[system]⚙ Running Pre-Flight Checks...[/system]")
    
    try:
        subprocess.run(["bwrap", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        console.print("[error]CRITICAL ERROR: Bubblewrap (bwrap) is not installed.[/error]")
        console.print("Please install it: sudo pacman -S bubblewrap")
        sys.exit(1)
        
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
    except Exception:
        console.print("[error]CRITICAL ERROR: Ollama is not running.[/error]")
        console.print("Please start it: systemctl start ollama")
        sys.exit(1)
        
    console.print("[success]✓ Pre-Flight Checks Passed.[/success]")
    console.print("[warning]⚠ Sandbox Active. The '~' (Home) path is mapped to a temporary RAM disk. Use relative paths ('./') to edit files in your actual directory.[/warning]")

def settings_menu():
    console.print("\n[bold green]=== Settings ===[/bold green]")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')[1:]
        models = [line.split()[0] for line in lines if line]
    except Exception:
        models = []
        
    if not models:
        console.print("[error]No Ollama models found! Please pull one (e.g. `ollama pull qwen3.5:4b`).[/error]")
        return
        
    table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
    table.add_column("ID", style="dim", justify="center")
    table.add_column("Local Model", style="bright_cyan")
    for idx, model in enumerate(models):
        table.add_row(str(idx), model)
    console.print(table)
        
    choice = Prompt.ask("Select Model", choices=[str(i) for i in range(len(models))])
    selected_model = models[int(choice)]
    
    stateful = Prompt.ask("Enable Conversational History for Local AI? (Warning: Slower and prone to hallucination on small models)", choices=["y", "n"]) == "y"
    require_conf = Prompt.ask("Require confirmation before executing commands?", choices=["y", "n"]) == "y"
    
    config = load_config()
    current_key = config.get("cloud_api_key", "")
    key_status = "[bold green]Configured[/bold green]" if current_key else "[dim red]Not Configured[/dim red]"
    
    console.print(f"\n[bold green]=== Cloud Provider Setup (Currently: {config.get('cloud_provider', 'google')}) ===[/bold green]")
    console.print(f"Current API Key Status: {key_status}")
    new_key = Prompt.ask("Enter new API Key (Leave blank to keep current)", password=True)
    if new_key.strip():
        config["cloud_api_key"] = new_key.strip()
        
    config["local_model"] = selected_model
    config["stateful_local"] = stateful
    config["require_confirmation"] = require_conf
    save_config(config)
    
    # Reload the agent with the new config dynamically
    reload_agent()
    console.print("[success]Settings Saved & Agent Reloaded![/success]")

def execute_graph(app, config, user_input):
    """Runs the graph for a given user input and prints the output."""
    console.print(Rule(style="info"))
    
    new_inputs = {
        "messages": [HumanMessage(content=user_input)], 
        "input_text": user_input,
        "retry_count": 0, 
        "error": None
    }
    
    result = app.invoke(new_inputs, config=config)
    
    final_output = result.get("command_output", "")
    if final_output:
        escaped_out = escape(final_output)
        console.print(Panel(escaped_out, title="[bold green]✓ Execution Output[/bold green]", border_style="success"))

def main():
    console.print(Panel("[bold cyan]DHI: Hybrid OS Agent[/bold cyan]", expand=False, border_style="cyan"))
    
    # Initialize the LLMs
    reload_agent()
    
    startup_checks()
    
    # Setup XDG paths
    data_dir = os.path.expanduser("~/.local/share/dhi")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "pragma_state.db")
    
    # Initialize Storage & Memory for persistent sessions
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # Compile the Graph WITH the Checkpointer attached
    app = workflow.compile(checkpointer=memory)

    # Define the Workspace Session
    config = {"configurable": {"thread_id": "main_workspace"}}

    # ONE-SHOT CLI MODE
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        try:
            execute_graph(app, config, user_input)
        finally:
            conn.close()
        return

    # INTERACTIVE REPL MODE
    try:
        while True:
            console.print()
            mode = Prompt.ask("[prompt]❯ Select Mode [dim](\\[t]ext, \\[v]oice, \\[s]ettings, \\[q]uit)[/dim][/prompt]", choices=["t", "v", "s", "q"], default="t")
            
            user_input = ""
            
            if mode == 'q':
                console.print("[muted]Shutting down. Session state saved to disk.[/muted]")
                break
                
            elif mode == 's':
                settings_menu()
                continue
                
            elif mode == 'v':
                console.print("[system]Loading Whisper Model...[/system]")
                ear = Ear(model_size="distil-small.en")
                    
                user_input = ear.listen_and_transcribe()
                
                # Unload Whisper immediately to free VRAM
                del ear
                import gc
                gc.collect()
                
                if not user_input:
                    continue
                    
                console.print(f"[bold magenta]Transcription:[/bold magenta] {escape(user_input)}")
                    
            elif mode == 't':
                user_input = Prompt.ask("[prompt]❯ Command[/prompt]")
            
            else:
                continue

            execute_graph(app, config, user_input)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
