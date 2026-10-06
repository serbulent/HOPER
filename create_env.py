"""Kept for backwards compatibility: environment creation now lives in create_env.sh."""
import os
import subprocess
import sys

script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_env.sh")
sys.exit(subprocess.call(["bash", script]))
