import os
import numpy as np
from langchain_ollama import OllamaEmbeddings
from dhi.ui import console

# XDG-compliant cache path
CACHE_DIR = os.path.expanduser("~/.cache/dhi")
COMPLEX_CACHE = os.path.join(CACHE_DIR, "anchors_complex.npy")
SIMPLE_CACHE = os.path.join(CACHE_DIR, "anchors_simple.npy")
CACHE_VERSION_FILE = os.path.join(CACHE_DIR, "anchors.version")

# Bump this when anchors change — forces a re-embed
ANCHOR_VERSION = "3"

class Router:
    def __init__(self):
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")

        # --- Expanded Anchor Sets ---
        # Canonical "complex" tasks (multi-step, piping, scripting, conversion)
        self.complex_anchors = [
            # Software Engineering & Scripting
            "write a python script to scrape a website",
            "refactor this entire codebase",
            "create a complex regex pattern for emails",
            "write a bash script that monitors disk usage and sends an alert",
            "create a virtual environment and install dependencies from requirements",
            "write a makefile for this c project",
            
            # Web Development & Frontend
            "generate a complex html webpage with javascript and css",
            "build a responsive react application dashboard",
            "write a full stack web application with a nodejs backend",
            "create a beautiful ui component with tailwind css",
            
            # Advanced Data & Document Processing
            "parse this json file and extract all email addresses",
            "use awk to process this csv and calculate column averages",
            "read this large text file and summarize the key points",
            "convert word documents to pdf using pandoc",
            "build a static site from markdown files using pandoc",
            "merge multiple pdf files into one document",
            
            # Media & Networking
            "convert this video to mp4 using ffmpeg",
            "transcode audio files from flac to opus using ffmpeg",
            "download all images from a webpage using curl and grep",
            "set up an ssh tunnel to forward a remote port",
            
            # Advanced System Administration
            "setup a docker container with nginx",
            "set up a cron job to backup my database every night",
            "create a systemd service file for my application",
            "use sed to find and replace across multiple files recursively",
            "find all duplicate files in this directory recursively",
            "compress this entire folder into a tar.gz archive",
            "manipulate windows in hyprland",
        ]

        # Canonical "simple" tasks (single command, direct output)
        self.simple_anchors = [
            # Basic File Operations
            "create a file called test.txt",
            "list all files in this directory",
            "make a new folder",
            "display the contents of a file",
            "count the number of lines in a file",
            "rename a file",
            "search for a file by name",
            "display the last 10 lines of a log file",
            
            # System Information
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
            
            # Basic Networking & Hardware Control
            "check if a port is open",
            "ping a server",
            "print hello world",
            "kill a process by name",
            "turn off wifi",
            "mute the system volume",
            "increase screen brightness",
        ]

        # Load or compute embeddings
        self.complex_embeddings = self._load_or_compute()

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
        """Load cached anchor embeddings from disk, or compute and cache them."""
        if self._cache_is_valid():
            console.print("[muted]ℹ Loading cached semantic anchors...[/muted]")
            self.complex_embeddings = np.load(COMPLEX_CACHE).tolist()
            self.simple_embeddings = np.load(SIMPLE_CACHE).tolist()
            return self.complex_embeddings

        console.print("[info]ℹ Computing semantic anchors (first run or anchors updated)...[/info]")
        self.complex_embeddings = self.embedder.embed_documents(self.complex_anchors)
        self.simple_embeddings = self.embedder.embed_documents(self.simple_anchors)

        # Cache to disk
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(COMPLEX_CACHE, np.array(self.complex_embeddings))
        np.save(SIMPLE_CACHE, np.array(self.simple_embeddings))
        with open(CACHE_VERSION_FILE, 'w') as f:
            f.write(ANCHOR_VERSION)

        console.print("[success]✓ Anchors cached to disk.[/success]")
        return self.complex_embeddings

    def _cosine_similarity(self, a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def route(self, input_text: str) -> dict:
        """Scores the input and returns a route with confidence.

        Returns:
            dict with keys:
                route: "local" or "cloud"
                confidence: float 0.0 - 1.0 (how sure the router is)
        """

        # 1. Embed the user input
        input_emb = self.embedder.embed_query(input_text)

        # 2. Score against both anchor sets
        complex_scores = sorted(
            [self._cosine_similarity(input_emb, anchor) for anchor in self.complex_embeddings],
            reverse=True
        )
        simple_scores = sorted(
            [self._cosine_similarity(input_emb, anchor) for anchor in self.simple_embeddings],
            reverse=True
        )

        # 3. Top-K mean (K=3) — more robust than single-max
        k = 3
        complex_score = float(np.mean(complex_scores[:k]))
        simple_score = float(np.mean(simple_scores[:k]))

        console.print(
            f"[muted]ℹ Semantic Match -> Complex: {complex_score:.3f} "
            f"(top: {complex_scores[0]:.3f}) | Simple: {simple_score:.3f} "
            f"(top: {simple_scores[0]:.3f})[/muted]"
        )

        # 4. Margin-based decision
        #    Route to cloud only if complex wins by a clear margin
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
