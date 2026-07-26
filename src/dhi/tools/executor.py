import subprocess
import os
import shutil
from rich.markup import escape
from dhi.ui import console

class SafeExecutor:
    def __init__(self, allowed_workdir=None):
        """Initialize the sandbox with Hyprland support."""
        self.workdir = allowed_workdir if allowed_workdir else os.getcwd()
        self.workdir = os.path.abspath(self.workdir)
        
        # Detect Hyprland signature.
        self.hypr_sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
        
        console.print("[info]ℹ Sandbox Initialized[/info]")
        if self.hypr_sig:
             console.print("[system]✦ Hyprland IPC Detected[/system]")

    def execute(self, command, requires_network=False):


        # Build core bubblewrap command.
        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/", "/",          # Read-only system
            "--dev", "/dev",                # Devices
            "--proc", "/proc",              # Processes
            "--bind", self.workdir, self.workdir, # Project write access
            "--tmpfs", "/tmp",              # Fake /tmp for safety
            
            # Direct applications to use the temporary /tmp folder for caches
            # instead of the read-only /home directory.
            "--setenv", "HOME", "/tmp",
            
            "--die-with-parent",
            "--new-session"
        ]
        
        # Zero-trust network sandboxing.
        if not requires_network:
            bwrap_cmd.append("--unshare-net")

        # Expose the Hyprland socket to control windows if detected.
        if self.hypr_sig:
            # Pass the environment variable.
            bwrap_cmd.extend(["--setenv", "HYPRLAND_INSTANCE_SIGNATURE", self.hypr_sig])
            
            # Bind the socket folder checking common locations.
            paths_to_check = [
                f"/tmp/hypr/{self.hypr_sig}",
                f"/run/user/{os.getuid()}/hypr/{self.hypr_sig}"
            ]
            
            for path in paths_to_check:
                if os.path.exists(path):
                    # Bind the specific socket directory to itself to bypass bwrap tmpfs complexity.
                    bwrap_cmd.extend(["--bind", path, path])

        # Add the target command.
        bwrap_cmd.extend(["bash", "-c", command])

        try:
            result = subprocess.run(
                bwrap_cmd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=60
            )
            
            if result.returncode == 0:
                stdout_text = result.stdout.strip()
                if "Error:" in stdout_text:
                    console.print("[warning]⚠ Script executed (Exit 0), but output contains an Error.[/warning]")
                else:
                    console.print("[success]✓ Execution Successful[/success]")
                
                final_out = stdout_text if stdout_text else "Command executed successfully with no output."
                return {"success": True, "output": final_out}
            else:
                error_msg = result.stderr.strip()
                # Handle standard errors (e.g., syntax error, missing package).
                if error_msg:
                    console.print(f"[error]⨯ Execution Failed (Exit Code {result.returncode})[/error]")
                    return {"success": False, "output": f"Error: {error_msg}"}
                # Handle silent failures (e.g., empty grep search).
                else:
                    console.print(f"[warning]⚠ Silent Exit (Code {result.returncode}) - No matches found.[/warning]")
                    return {"success": False, "output": "Command executed, but returned no matches or output."}

        except subprocess.TimeoutExpired:
            console.print("[error]⨯ Command Timed Out[/error]")
            return {"success": False, "output": "Error: Execution timed out."}
        except Exception as e:
            return {"success": False, "output": f"Error: {e}"}
