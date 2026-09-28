"""Sanity check for the project setup. Run: python src/check_setup.py

Checks that the required packages import and that the API key is loaded
from .env. It never prints the key itself and makes no API calls.
"""
import os
import sys

from dotenv import load_dotenv

print(f"Python {sys.version.split()[0]}")

for package in ["anthropic", "pandas", "sklearn", "matplotlib"]:
    module = __import__(package)
    print(f"  OK  {package} {module.__version__}")

load_dotenv()  # reads .env in the current folder into environment variables
key = os.getenv("ANTHROPIC_API_KEY")
if key and key.startswith("sk-ant-"):
    print("  OK  ANTHROPIC_API_KEY found in .env")
else:
    print("  MISSING  ANTHROPIC_API_KEY - copy .env.example to .env and add your key")
    sys.exit(1)
