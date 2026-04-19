import platform
import os
import sys

print("System Information:")
print(f"OS Name: {platform.system()}")
print(f"Node Name: {platform.node()}")
print(f"Release: {platform.release()}")
print(f"Version: {platform.version()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print(f"Python Version: {sys.version}")
print(f"Current Working Directory: {os.getcwd()}")
