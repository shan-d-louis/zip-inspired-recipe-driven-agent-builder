"""Prompt text shared by the engine and the scripted (fake) model.

Step 4 builds the full prompts here. The marker below is what tells any model,
real or scripted, that the final answer must be Findings JSON.
"""

JSON_INSTRUCTION = "Return only a JSON object that matches this schema:"
