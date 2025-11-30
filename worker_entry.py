"""Railway entrypoint for running the ARQ worker from root directory.

This file is called by Railway's worker process configuration.
It imports and runs the actual worker settings from the background_jobs module.
"""

import sys
import os

# Add app directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arq import run_worker
from background_jobs.arq_worker import WorkerSettings


if __name__ == "__main__":
    run_worker(WorkerSettings)
