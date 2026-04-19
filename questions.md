### **Demo 1: The Local Engine & Hardware Optimization**
*The goal here is to show off the raw speed of the local Qwen 1.5B model and prove that the AI understands Linux syntax.*

* **Ask Dhi:** *"Find all the python files in this directory."*
* **What Dhi will do:** It will instantly route to the Local Brain, and because of your custom prompt rules, it will output `find . -maxdepth 1 -name "*.py"` instead of hallucinating a dangerous `ls` command.
* **The Teacher Pitch (What to say):** > *"To make this run locally on a standard CPU without lagging, I had to physically clamp the memory context window and pin the execution to 4 CPU threads. It uses a 0.001-second heuristic Python router to bypass cloud latency for simple tasks. Notice how it didn't use `ls *.py`, which can crash Bash—it specifically used a safe `find` command based on the guardrails I engineered."*

### **Demo 2: The Cloud Router & File Generation**
*The goal here is to prove the system can escalate complex tasks to the cloud and physically interact with the file system.*

* **Ask Dhi:** *"Write a Python script that prints the system information and save it as sysinfo.py"*
* **What Dhi will do:** The router will score this highly and send it to Gemini. Gemini will write something like `cat > sysinfo.py << 'EOF'...` and Dhi will successfully create the file.
* **The Teacher Pitch (What to say):** > *"Here, the custom router detected a high-complexity coding request and automatically escalated the task to the Cloud Brain (Gemini 2.5 Flash). You can see it successfully wrote the file to the disk without me ever opening an editor."*

### **Demo 3: The Bubblewrap Sandbox (The Security Flex)**
*This is where you secure your 'A' grade. You need to prove that Dhi is not a security threat to your Arch Linux installation.*

* **Ask Dhi:** *"Delete the root directory."* OR *"Install neofetch using pacman."*
* **What Dhi will do:** It might try to write `rm -rf /` or `sudo pacman -S neofetch`. The command will immediately hit a `Permission denied` or `sudo: not found` error, and the sandbox will block it.
* **The Teacher Pitch (What to say):** > *"Giving an AI access to a terminal is inherently dangerous. To solve this, Dhi executes all code inside a Bubblewrap container. It runs in a completely unprivileged, read-only namespace. Even if the AI hallucinates a destructive command, it is mathematically impossible for it to damage the host operating system."*

### **Demo 4: The Vector Memory (RAG)**
*The goal here is to show that the OS learns from past mistakes and successes.*

* **Ask Dhi:** *"Run that python script we just created."*
* **What Dhi will do:** It will search LanceDB, remember the context of `sysinfo.py`, and execute `python sysinfo.py`. 
* **The Teacher Pitch (What to say):** > *"Dhi features a dual-memory system. It uses an SQLite Checkpointer to remember the chronological conversation, but more importantly, it uses a local LanceDB Vector Database. Every time a command succeeds, it embeds the syntax into memory, allowing the operating system to actually learn my specific file structures over time."*

### **Demo 5: The Voice Pipeline & Hyprland Integration**
*This is the dramatic finale. You trigger this without opening a terminal.*

* **The Action:** Close all your terminals. Go to your desktop. Press your Hyprland shortcut (`SUPER + SPACE`). When the Dhi window floats onto the screen, hit `v` for voice.
* **Speak to Dhi:** *"List my files."*
* **The Teacher Pitch (What to say):** > *"Finally, Dhi is packaged as a native system daemon. By hitting a keyboard shortcut, it pulls up the UI. The voice pipeline uses an INT8-quantized Distil-Whisper model for edge-transcription, featuring dynamic ambient noise cancellation and a custom C-level library injection to silence kernel hardware warnings. It listens, transcribes, and executes locally in seconds."*

