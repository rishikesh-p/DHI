import time
import lancedb
import os
from langchain_ollama import OllamaEmbeddings
from dhi.ui import console

# Threshold for cosine distance to filter dissimilar results.
# 0 = identical, 1 = orthogonal, 2 = opposite.
RELEVANCE_THRESHOLD = 0.5

# Threshold for deduplication to prevent saving near-identical entries.
DEDUP_THRESHOLD = 0.1

class MemorySystem:
    def __init__(self, db_path=None):
        """Initialize LanceDB with Ollama Embeddings."""
        if db_path is None:
            db_path = os.path.expanduser("~/.local/share/dhi/lancedb")
            
        console.print("[info]ℹ Initializing Vector DB...[/info]")
        
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")
        
        try:
            self.db = lancedb.connect(db_path)
            
            # Attempt to open the existing table
            try:
                self.table = self.db.open_table("knowledge_base")
                
                # Verify schema backward compatibility
                expected_columns = {"intent", "command", "vector", "timestamp", "success"}
                existing_columns = set(self.table.schema.names)
                
                if not expected_columns.issubset(existing_columns):
                    console.print("[warning]⚠ Outdated Memory DB schema detected. Upgrading...[/warning]")
                    self.db.drop_table("knowledge_base")
                    raise ValueError("Schema upgrade required")
                    
                console.print("[success]✓ Loaded existing database.[/success]")
            except:
                # Create table with schema if open fails
                console.print("[warning]⚠ Creating new knowledge base...[/warning]")
                seed_vec = self.embedder.embed_query("__init__")
                data = [{
                    "intent": "__init__",
                    "command": "__init__",
                    "vector": seed_vec,
                    "timestamp": time.time(),
                    "success": True
                }]
                self.table = self.db.create_table("knowledge_base", data)
                
        except Exception as e:
            console.print(f"[error]⨯ Critical DB Error: {e}[/error]")
            self.table = None

    def save(self, intent, command, success=True):
        """Store a text snippet in long-term memory.
        
        Skip saving if a near-duplicate already exists based on the deduplication threshold.
        """
        if not self.table: return
        
        # Convert text to vector
        vector = self.embedder.embed_query(intent)
        
        # Check if a similar entry already exists to avoid duplicates
        try:
            existing = self.table.search(vector).distance_type("cosine").limit(1).to_pandas()
            if not existing.empty and existing.iloc[0]['_distance'] < DEDUP_THRESHOLD:
                console.print(f"[muted]ℹ Skipping save — similar entry already exists.[/muted]")
                return
        except Exception:
            pass  # If search fails, save anyway
        
        # Add to database with metadata
        self.table.add([{
            "intent": intent,
            "command": command,
            "vector": vector,
            "timestamp": time.time(),
            "success": success
        }])
        console.print(f"[muted]ℹ Stored in memory.[/muted]")

    def recall(self, query, limit=3):
        """Find the most relevant memories for a query.
        
        Return up to `limit` results filtered by relevance threshold, excluding the seed record.
        """
        if not self.table: return ""
        
        # Convert query to vector
        query_vec = self.embedder.embed_query(query)
        
        results = self.table.search(query_vec).distance_type("cosine").limit(limit + 1).to_pandas()
        
        if results.empty:
            return ""
        
        # Remove seed record and apply relevance threshold
        results = results[results['intent'] != "__init__"]
        results = results[results['_distance'] < RELEVANCE_THRESHOLD]
        
        if results.empty:
            return ""
        
        # Take top `limit` results
        results = results.head(limit)
        
        # Format results as a context string
        matches = [f"Request: {row['intent']} -> Command: {row['command']}" for _, row in results.iterrows()]
        context = "\n".join(f"- {m}" for m in matches)
        
        console.print(f"[system]✦ Recalled {len(matches)} memory(s)[/system]")
        return context

    def exact_match(self, query, threshold=0.05):
        """Check if the query matches a previous intent almost exactly.
        
        Return the exact command to bypass the LLM if the distance is below the threshold.
        """
        if not self.table: return None
        
        query_vec = self.embedder.embed_query(query)
        try:
            results = self.table.search(query_vec).distance_type("cosine").limit(2).to_pandas()
            for _, row in results.iterrows():
                if row['intent'] == "__init__": continue
                if row['_distance'] < threshold:
                    return row['command']
            return None
        except Exception:
            return None

# Unit Tests
if __name__ == "__main__":
    mem = MemorySystem()
    
    # Test 1: Teach it something
    print("\n--- Teaching ---")
    mem.save("list files", "ls -la")
    mem.save("show disk usage", "df -h")
    
    # Test 2: Try saving a duplicate
    print("\n--- Duplicate ---")
    mem.save("list files", "ls -la")  # Should skip
    
    # Test 3: Ask it something
    print("\n--- Asking ---")
    context = mem.recall("list all my files")
    print(f"Context found:\n{context}")
