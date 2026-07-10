import sys
import os

# Add .agents/helpers to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".agents", "helpers")))
from gemma_reasoner import query_gemma

with open("reasoning_prompt_tw.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

print("Querying Llama-3.3-70b-versatile via Groq...")
res = query_gemma(prompt)

with open("gemma_reasoner_out.txt", "w", encoding="utf-8") as f:
    f.write(res)

print("Reasoning completed. Result saved to gemma_reasoner_out.txt")
