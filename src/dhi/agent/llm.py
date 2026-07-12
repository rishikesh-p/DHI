import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dhi.ui import console
from dhi.config import load_config

# Load environment variables
load_dotenv()

class AI_Brain:
    def __init__(self, mode="local", temperature=0):
        """Initialize the AI brain.
        
        Args:
            mode: 'local' (Ollama) or 'cloud' (Gemini)
            temperature: The temperature for the LLM.
        """
        self.mode = mode
        
        if mode == "cloud":
            config = load_config()
            provider = config.get("cloud_provider", "google")
            api_key = config.get("cloud_api_key", "")
            
            if not api_key:
                # Fallback to .env configuration
                api_key = os.getenv("GOOGLE_API_KEY")
                
            if not api_key:
                console.print("[error]⨯ Cloud API Key not configured! Use the [s] Settings Menu to set it.[/error]")
                raise ValueError("Missing API Key")
            
            if provider == "google":
                model_name = config.get("cloud_model", "gemini-2.5-flash")
                console.print(f"[info]ℹ Connecting to Cloud ({model_name})...[/info]")
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key
                )
            else:
                console.print(f"[error]⨯ Unsupported cloud provider: {provider}[/error]")
                raise ValueError("Unsupported Cloud Provider")
        else:
            # Configure local model
            config = load_config()
            model_name = config.get("local_model", "qwen3.5:4b")
            console.print(f"[info]ℹ Connecting to Local ({model_name})...[/info]")
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                stop=["<|endoftext|>", "User:", "<|im_end|>", "<|im_start|>"], 
                reasoning=False,
                keep_alive="30m",
                max_tokens=100,
                num_ctx=2048,
                num_thread=4
            )

    def think(self, prompt, system_prompt=None):
        """Send a prompt to the selected LLM and stream the response.
        
        Args:
            prompt: The user input prompt.
            system_prompt: Behavioral instructions for the LLM. Required.
        """
        if not system_prompt:
            raise ValueError("system_prompt is required. Pass LOCAL_SYSTEM_PROMPT or CLOUD_SYSTEM_PROMPT from graph.py.")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            # Stream tokens in real-time
            full_response = ""
            chunks = self.llm.stream(messages)
            
            with console.status(f"[bold cyan]Brain ({self.mode.upper()}) is evaluating prompt...[/bold cyan]", spinner="dots"):
                first_chunk = next(chunks, None)
                
            if first_chunk:
                console.print(first_chunk.content, style="cyan", end="", markup=False)
                full_response += first_chunk.content
                for chunk in chunks:
                    console.print(chunk.content, style="cyan", end="", markup=False)
                    full_response += chunk.content
            
            print()
            return full_response
        except Exception as e:
            return f"Error during thought process: {e}"

# --- Unit Test ---
if __name__ == "__main__":
    test_prompt = "You are a test assistant. Output only the exact text requested."
    
    # Test Local
    print("\n--- Testing Local ---")
    local_brain = AI_Brain(mode="local")
    print(local_brain.think("Say 'Local is working'", system_prompt=test_prompt))

    # Test Cloud
    print("\n--- Testing Cloud ---")
    try:
        cloud_brain = AI_Brain(mode="cloud")
        print(cloud_brain.think("Say 'Cloud is working'", system_prompt=test_prompt))
    except Exception as e:
        print(f"Cloud Test Failed: {e}")
