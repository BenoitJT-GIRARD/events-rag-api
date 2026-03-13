import json

import httpx


BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    with httpx.Client(timeout=60.0) as client:
        health = client.get(f"{BASE_URL}/health")
        metadata = client.get(f"{BASE_URL}/metadata")
        ask = client.post(
            f"{BASE_URL}/ask",
            json={
                "question": "Quels événements musicaux ont lieu à Montpellier ?",
                "top_k": 5,
            },
        )

    print("HEALTH")
    print(json.dumps(health.json(), ensure_ascii=False, indent=2))

    print("\nMETADATA")
    print(json.dumps(metadata.json(), ensure_ascii=False, indent=2))

    print("\nASK")
    print(json.dumps(ask.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()