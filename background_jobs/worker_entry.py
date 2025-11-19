"""Railway entrypoint for running the ARQ worker.

This keeps process startup simple so Railway can run:
    python worker_entry.py
"""

from arq import run_worker

from arq_worker import WorkerSettings


if __name__ == "__main__":
    run_worker(WorkerSettings)
