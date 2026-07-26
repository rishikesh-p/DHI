import os
import numpy as np
from langchain_ollama import OllamaEmbeddings
from dhi.ui import console

# Set XDG-compliant cache path.
CACHE_DIR = os.path.expanduser("~/.cache/dhi")
COMPLEX_CACHE = os.path.join(CACHE_DIR, "anchors_complex.npy")
SIMPLE_CACHE = os.path.join(CACHE_DIR, "anchors_simple.npy")
CACHE_VERSION_FILE = os.path.join(CACHE_DIR, "anchors.version")

# Version counter to force re-embed when anchors change.
ANCHOR_VERSION = "3"

class Router:
    def __init__(self):
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")

        # Define complex anchors for multi-step tasks, piping, and scripting.
        self.complex_anchors = [
            # Software engineering and scripting
            "write a python script to scrape a website",
            "refactor this entire codebase",
            "create a complex regex pattern for emails",
            "write a bash script that monitors disk usage and sends an alert",
            "create a virtual environment and install dependencies from requirements",
            "write a makefile for this c project",
            
            # Web development and frontend
            "generate a complex html webpage with javascript and css",
            "build a responsive react application dashboard",
            "write a full stack web application with a nodejs backend",
            "create a beautiful ui component with tailwind css",
            
            # Advanced data and document processing
            "parse this json file and extract all email addresses",
            "use awk to process this csv and calculate column averages",
            "read this large text file and summarize the key points",
            "convert word documents to pdf using pandoc",
            "build a static site from markdown files using pandoc",
            "merge multiple pdf files into one document",
            
            # Media and networking
            "convert this video to mp4 using ffmpeg",
            "transcode audio files from flac to opus using ffmpeg",
            "download all images from a webpage using curl and grep",
            "set up an ssh tunnel to forward a remote port",
            
            # Advanced system administration
            "setup a docker container with nginx",
            "set up a cron job to backup my database every night",
            "create a systemd service file for my application",
            "use sed to find and replace across multiple files recursively",
            "find all duplicate files in this directory recursively",
            "compress this entire folder into a tar.gz archive",
            "manipulate windows in hyprland",
        ]

        # Define simple anchors for single command execution.
        self.simple_anchors = [
            # Basic file operations
            "create a file called test.txt",
            "list all files in this directory",
            "make a new folder",
            "display the contents of a file",
            "count the number of lines in a file",
            "rename a file",
            "search for a file by name",
            "display the last 10 lines of a log file",
            
            # System information
            "what is the current date",
            "tell me my ip address",
            "show me running processes",
            "check how much disk space is left",
            "show the current working directory",
            "show environment variables",
            "check the linux kernel version",
            "show memory usage",
            "show system uptime",
            "display the current user",
            
            # Basic networking and hardware control
            "check if a port is open",
            "ping a server",
            "print hello world",
            "kill a process by name",
            "turn off wifi",
            "mute the system volume",
            "increase screen brightness",
        ]

        # Load or compute embeddings as Numpy matrices.
        self._load_or_compute()

    def _cache_is_valid(self):
        """Check if cached embeddings exist and match the current anchor version."""
        if not all(os.path.exists(f) for f in [COMPLEX_CACHE, SIMPLE_CACHE, CACHE_VERSION_FILE]):
            return False
        try:
            with open(CACHE_VERSION_FILE, 'r') as f:
                return f.read().strip() == ANCHOR_VERSION
        except (IOError, OSError):
            return False

    def _load_or_compute(self):
        """Load cached anchor embeddings from disk, or compute and cache them.

        Store embeddings as normalized 2D Numpy arrays for vectorized scoring.
        """
        if self._cache_is_valid():
            console.print("[muted]ℹ Loading cached semantic anchors...[/muted]")
            self.complex_embeddings = np.load(COMPLEX_CACHE)
            self.simple_embeddings = np.load(SIMPLE_CACHE)
        else:
            console.print("[info]ℹ Computing semantic anchors (first run or anchors updated)...[/info]")
            self.complex_embeddings = np.array(self.embedder.embed_documents(self.complex_anchors))
            self.simple_embeddings = np.array(self.embedder.embed_documents(self.simple_anchors))

            # Cache embeddings to disk.
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.save(COMPLEX_CACHE, self.complex_embeddings)
            np.save(SIMPLE_CACHE, self.simple_embeddings)
            with open(CACHE_VERSION_FILE, 'w') as f:
                f.write(ANCHOR_VERSION)
            console.print("[success]✓ Anchors cached to disk.[/success]")

        # Pre-normalize rows for fast cosine similarity via dot product (add epsilon for safety).
        self._complex_normed = self.complex_embeddings / (np.linalg.norm(self.complex_embeddings, axis=1, keepdims=True) + 1e-10)
        self._simple_normed = self.simple_embeddings / (np.linalg.norm(self.simple_embeddings, axis=1, keepdims=True) + 1e-10)

    def route(self, input_text: str) -> dict:
        """Score the input and return a route with confidence."""
        input_emb = np.array(self.embedder.embed_query(input_text))
        return self.route_vec(input_emb)

    def route_vec(self, input_emb) -> dict:
        """Score the input from a pre-computed vector."""
        input_normed = np.array(input_emb) / (np.linalg.norm(input_emb) + 1e-10)

        # Vectorized cosine similarity via matrix-vector dot product.
        complex_scores = self._complex_normed @ input_normed
        simple_scores = self._simple_normed @ input_normed

        # Calculate Top-K mean to improve robustness.
        k = 3
        complex_score = float(np.sort(complex_scores)[-k:].mean())
        simple_score = float(np.sort(simple_scores)[-k:].mean())

        console.print(
            f"[muted]ℹ Semantic Match -> Complex: {complex_score:.3f} "
            f"(top: {complex_scores[0]:.3f}) | Simple: {simple_score:.3f} "
            f"(top: {simple_scores[0]:.3f})[/muted]"
        )

        # Make a margin-based decision.
        # Route to cloud only if complex wins by the specified margin.
        margin = complex_score - simple_score
        margin_threshold = 0.03  # Must win by at least this much

        if margin > margin_threshold:
            route = "cloud"
            confidence = min(1.0, margin / 0.15)  # Normalize: 0.15+ margin = full confidence
        else:
            route = "local"
            confidence = min(1.0, abs(margin) / 0.15) if margin < -margin_threshold else 0.5

        console.print(
            f"[muted]ℹ Decision: margin={margin:+.3f} -> {route.upper()} "
            f"(confidence: {confidence:.0%})[/muted]"
        )

        return {"route": route, "confidence": confidence}
