import subprocess
import os
import shutil
from rich.markup import escape
from dhi.ui import console

class SafeExecutor:
    def __init__(self, allowed_workdir=None):
        """
        Initializes the Sandbox with Hyprland Support.
        """
        self.workdir = allowed_workdir if allowed_workdir else os.getcwd()
        self.workdir = os.path.abspath(self.workdir)
        
        # Detect Hyprland Signature
        self.hypr_sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
        
        console.print("[info]ℹ Sandbox Initialized[/info]")
        if self.hypr_sig:
             console.print("[system]✦ Hyprland IPC Detected[/system]")

    def execute(self, command):
        # We don't print the execution here anymore because graph.py handles the UI logging for execution.
        # This keeps the output cleaner.

        # Core Bubblewrap Command
        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/", "/",          # Read-only System
            "--dev", "/dev",                # Devices
            "--proc", "/proc",              # Processes
            "--bind", self.workdir, self.workdir, # Project Write Access
            "--tmpfs", "/tmp",              # Fake /tmp for safety
            
            # --- THE GENERIC SOLUTION ---
            # Trick all messy applications into using the temporary /tmp 
            # folder for their caches instead of crashing against the Read-Only /home
            "--setenv", "HOME", "/tmp",
            # -------------------------------
            
            "--die-with-parent",
            "--new-session"
        ]

        # --- HYPRLAND INTEGRATION ---
        # If we are in Hyprland, we must expose the socket to control windows
        if self.hypr_sig:
            # 1. Pass the Env Variable
            bwrap_cmd.extend(["--setenv", "HYPRLAND_INSTANCE_SIGNATURE", self.hypr_sig])
            
            # 2. Bind the Socket Folder (Usually /tmp/hypr)
            # We check both common locations
            paths_to_check = [
                f"/tmp/hypr/{self.hypr_sig}",
                f"/run/user/{os.getuid()}/hypr/{self.hypr_sig}"
            ]
            
            for path in paths_to_check:
                if os.path.exists(path):
                    # We bind it to the SAME path inside the sandbox
                    # We need to make the parent dir first inside the tmpfs
                    # But bwrap is tricky. The easiest way is to bind the specific folder.
                    # Since we are overlaying a tmpfs on /tmp, we need to be careful.
                    
                    # Strategy: Bind the specific socket dir to itself
                    bwrap_cmd.extend(["--bind", path, path])

        # Add the command to run
        bwrap_cmd.extend(["bash", "-c", command])

        try:
            result = subprocess.run(
                bwrap_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                console.print("[success]✓ Execution Successful[/success]")
                return result.stdout.strip() if result.stdout.strip() else "Command executed successfully with no output."
            else:
                error_msg = result.stderr.strip()
                # If stderr has text, it's a real crash (syntax error, missing package, etc.)
                if error_msg:
                    console.print(f"[error]⨯ Execution Failed (Exit Code {result.returncode})[/error]")
                    return f"Error: {error_msg}"
                # If stderr is empty, it was a silent failure (like an empty grep search)
                else:
                    console.print(f"[warning]⚠ Silent Exit (Code {result.returncode}) - No matches found.[/warning]")
                    return "Command executed, but returned no matches or output."

        except subprocess.TimeoutExpired:
            console.print("[error]⨯ Command Timed Out[/error]")
            return "Error: Execution timed out."
        except Exception as e:
            return f"Error: {e}"
