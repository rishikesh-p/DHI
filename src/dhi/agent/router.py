import json
import numpy as np
from langchain_ollama import OllamaEmbeddings
from dhi.ui import console

class Router:
    def __init__(self):
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")
        
        # A list of canonical "complex" tasks
        self.complex_anchors = [
            "write a python script to scrape a website",
            "convert this video to mp4 using ffmpeg",
            "setup a docker container with nginx",
            "refactor this entire codebase",
            "create a complex regex pattern for emails",
            "manipulate windows in hyprland",
            "convert word documents to pdf using pandoc",
            "compress this entire folder into a tar.gz archive"
        ]
        
        # A list of canonical "simple" tasks
        self.simple_anchors = [
            "create a file called test.txt",
            "what is the current date",
            "list all files in this directory",
            "tell me my ip address",
            "print hello world",
            "make a new folder",
            "show me running processes"
        ]
        
        console.print("[info]ℹ Initializing semantic anchors...[/info]")
        # Pre-compute anchor embeddings
        self.complex_embeddings = self.embedder.embed_documents(self.complex_anchors)
        self.simple_embeddings = self.embedder.embed_documents(self.simple_anchors)

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def route(self, input_text: str) -> str:
        """Scores the input using semantic similarity to determine complexity."""
        
        # 1. Embed the user input
        input_emb = self.embedder.embed_query(input_text)
        
        # 2. Compare against complex anchors
        complex_scores = [self._cosine_similarity(input_emb, anchor) for anchor in self.complex_embeddings]
        max_complex_score = max(complex_scores)
        
        # 3. Compare against simple anchors
        simple_scores = [self._cosine_similarity(input_emb, anchor) for anchor in self.simple_embeddings]
        max_simple_score = max(simple_scores)
        
        console.print(f"[muted]ℹ Semantic Match -> Complex: {max_complex_score:.2f} | Simple: {max_simple_score:.2f}[/muted]")
        
        # 4. Decision: If it maps closer to a complex task, route to Cloud.
        if max_complex_score > max_simple_score and max_complex_score > 0.6:
            return "cloud"
            
        return "local"
