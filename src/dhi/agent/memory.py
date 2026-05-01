import lancedb
import os
from langchain_ollama import OllamaEmbeddings
from dhi.ui import console

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
                console.print("[success]✓ Loaded existing database.[/success]")
            except:
                # ATTEMPT 2: If open fails, create it
                console.print("[warning]⚠ Creating new knowledge base...[/warning]")
                data = [{"text": "Init", "vector": self.embedder.embed_query("Init")}]
                self.table = self.db.create_table("knowledge_base", data)
                
        except Exception as e:
            console.print(f"[error]⨯ Critical DB Error: {e}[/error]")
            self.table = None

    def save(self, text):
        """
        Store a text snippet in long-term memory.
        """
        if not self.table: return
        
        console.print(f"[info]ℹ Storing: '{text}'...[/info]")
        
        # 1. Convert text to vector
        vector = self.embedder.embed_query(text)
        
        # 2. Add to DB
        self.table.add([{"text": text, "vector": vector}])

    def recall(self, query, limit=1):
        """
        Find the most relevant memories for a query.
        """
        if not self.table: return ""
        
        # 1. Convert query to vector
        query_vec = self.embedder.embed_query(query)
        
        # 2. Search DB
        results = self.table.search(query_vec).limit(limit).to_pandas()
        
        if results.empty:
            return ""
        
        # Return the best match text
        best_match = results.iloc[0]['text']
        console.print(f"[system]✦ Recalled: {best_match}[/system]")
        return best_match

# --- Unit Test ---
if __name__ == "__main__":
    mem = MemorySystem()
    
    # Test 1: Teach it something
    print("\n--- Teaching ---")
    mem.save("The user's favorite editor is Neovim.")
    mem.save("The project is located in /home/rishikesh/linux-ai")
    
    # Test 2: Ask it something
    print("\n--- Asking ---")
    q = "What text editor do I use?"
    context = mem.recall(q)
    print(f"Context found: {context}")
