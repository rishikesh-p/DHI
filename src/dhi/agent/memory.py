import time
import lancedb
import os
from langchain_ollama import OllamaEmbeddings
from dhi.ui import console

# Distance threshold — discard results less similar than this
# We use Cosine Distance: 0 = identical, 1 = orthogonal, 2 = opposite
# 0.5 cosine distance ≈ 0.5 cosine similarity — a reasonable relevance cutoff
RELEVANCE_THRESHOLD = 0.5

# If a near-duplicate exists within this distance, skip saving
# 0.1 cosine distance ≈ 0.9 cosine similarity
DEDUP_THRESHOLD = 0.1

class MemorySystem:
    def __init__(self, db_path=None):
        """
        Initialize LanceDB with Ollama Embeddings.
        """
        if db_path is None:
            db_path = os.path.expanduser("~/.local/share/dhi/lancedb")
            
        console.print("[info]ℹ Initializing Vector DB...[/info]")
        
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")
        
        try:
            self.db = lancedb.connect(db_path)
            
            # ATTEMPT 1: Try to open the existing table
            try:
                self.table = self.db.open_table("knowledge_base")
                
                # Verify schema backward compatibility
                expected_columns = {"text", "vector", "timestamp", "success"}
                existing_columns = set(self.table.schema.names)
                
                if not expected_columns.issubset(existing_columns):
                    console.print("[warning]⚠ Outdated Memory DB schema detected. Upgrading...[/warning]")
                    self.db.drop_table("knowledge_base")
                    raise ValueError("Schema upgrade required")
                    
                console.print("[success]✓ Loaded existing database.[/success]")
            except:
                # ATTEMPT 2: If open fails, create it with the schema
                console.print("[warning]⚠ Creating new knowledge base...[/warning]")
                seed_vec = self.embedder.embed_query("__init__")
                data = [{
                    "text": "__init__",
                    "vector": seed_vec,
                    "timestamp": time.time(),
                    "success": True
                }]
                self.table = self.db.create_table("knowledge_base", data)
                
        except Exception as e:
            console.print(f"[error]⨯ Critical DB Error: {e}[/error]")
            self.table = None

    def save(self, text, success=True):
        """
        Store a text snippet in long-term memory.
        Skips saving if a near-duplicate already exists (deduplication).
        """
        if not self.table: return
        
        # 1. Convert text to vector
        vector = self.embedder.embed_query(text)
        
        # 2. Deduplicate — check if a very similar entry already exists
        try:
            existing = self.table.search(vector).distance_type("cosine").limit(1).to_pandas()
            if not existing.empty and existing.iloc[0]['_distance'] < DEDUP_THRESHOLD:
                console.print(f"[muted]ℹ Skipping save — similar entry already exists.[/muted]")
                return
        except Exception:
            pass  # If search fails, save anyway
        
        # 3. Add to DB with metadata
        self.table.add([{
            "text": text,
            "vector": vector,
            "timestamp": time.time(),
            "success": success
        }])
        console.print(f"[muted]ℹ Stored in memory.[/muted]")

    def recall(self, query, limit=3):
        """
        Find the most relevant memories for a query.
        Returns up to `limit` results, filtered by relevance threshold.
        Excludes the __init__ seed record.
        """
        if not self.table: return ""
        
        # 1. Convert query to vector
        query_vec = self.embedder.embed_query(query)
        
        # 2. Search DB
        results = self.table.search(query_vec).distance_type("cosine").limit(limit + 1).to_pandas()
        
        if results.empty:
            return ""
        
        # 3. Filter: remove seed record, apply relevance threshold
        results = results[results['text'] != "__init__"]
        results = results[results['_distance'] < RELEVANCE_THRESHOLD]
        
        if results.empty:
            return ""
        
        # 4. Take top `limit` results
        results = results.head(limit)
        
        # 5. Format as context string
        matches = results['text'].tolist()
        context = "\n".join(f"- {m}" for m in matches)
        
        console.print(f"[system]✦ Recalled {len(matches)} memory(s)[/system]")
        return context

    def exact_match(self, query, threshold=0.05):
        """
        Check if the query matches a previous intent almost exactly.
        If distance is < 0.05, we short-circuit the LLM and return the exact command.
        """
        if not self.table: return None
        
        query_vec = self.embedder.embed_query(query)
        try:
            results = self.table.search(query_vec).distance_type("cosine").limit(2).to_pandas()
            for _, row in results.iterrows():
                if row['text'] == "__init__": continue
                if row['_distance'] < threshold:
                    full_text = row['text']
                    if " -> Command: " in full_text:
                        cmd = full_text.split(" -> Command: ", 1)[1]
                        return cmd
            return None
        except Exception:
            return None

# --- Unit Test ---
if __name__ == "__main__":
    mem = MemorySystem()
    
    # Test 1: Teach it something
    print("\n--- Teaching ---")
    mem.save("Request: list files -> Command: ls -la")
    mem.save("Request: show disk usage -> Command: df -h")
    
    # Test 2: Try saving a duplicate
    print("\n--- Duplicate ---")
    mem.save("Request: list files -> Command: ls -la")  # Should skip
    
    # Test 3: Ask it something
    print("\n--- Asking ---")
    context = mem.recall("list all my files")
    print(f"Context found:\n{context}")
