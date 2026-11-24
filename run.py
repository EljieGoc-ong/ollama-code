"""Run ollama-code without installing it: python /path/to/ollama-code/run.py (from your project folder)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ollama_code.cli import main  # noqa: E402

sys.exit(main())
