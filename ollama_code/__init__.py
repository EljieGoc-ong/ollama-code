"""ollama-code: a small Claude-Code-style coding agent for your terminal, powered by Ollama.

It talks to Ollama's /api/chat endpoint with tool calling, so the model can read, search,
edit files and run commands in your project, asking you before anything that changes state.
Standard library only; runs on Windows and Linux (and macOS).
"""

from .cli import main  # noqa: E402,F401  (entry point; older launchers import it from here)
