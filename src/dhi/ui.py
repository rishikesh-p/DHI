from rich.console import Console
from rich.theme import Theme

dhi_theme = Theme({
    "info": "blue",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "system": "magenta",
    "prompt": "bold cyan",
    "muted": "dim"
})

console = Console(theme=dhi_theme)
