import subprocess
import sys

COMMANDS = [
    ["uv", "run", "python", "scripts/build_dataset.py"],
    ["uv", "run", "python", "scripts/build_index.py"],
    ["uv", "run", "python", "scripts/evaluate_rag.py"],
    ["uv", "run", "python", "scripts/evaluate_ragas.py"],
    [
        "uv",
        "run",
        "uvicorn",
        "events_rag.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ],
]


def main() -> None:
    for command in COMMANDS[:-1]:
        print(f"\n>>> Running: {' '.join(command)}")
        subprocess.run(command, check=True)

    print(f"\n>>> Starting API: {' '.join(COMMANDS[-1])}")
    subprocess.run(COMMANDS[-1], check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)
