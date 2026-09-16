import os
import sys
import json
import time
import hashlib
import requests

def main():
    server_ip = os.environ.get("SERVER_TAILSCALE_IP", "").strip()
    if not server_ip or server_ip == "100.x.y.z":
        print("CRITICAL ERROR: SERVER_TAILSCALE_IP environment variable is unset, empty, or unconfigured!")
        print("Please set SERVER_TAILSCALE_IP in your environment or .env file before running.")
        sys.exit(1)

    server_port = os.environ.get("SERVER_PORT", "8080").strip()
    client_id = os.environ.get("CLIENT_ID", "1").strip()

    # Search for weights file
    weights_path = None
    candidate_weights = [
        f"/app/weights/client{client_id}_weights.pt",
        f"/app/client{client_id}_weights.pt",
        f"/app/weights/client_{client_id}_weights.pt",
        f"/app/client_{client_id}_weights.pt",
        "/app/client_weights.pt",
        "/app/weights.pt"
    ]
    for cand in candidate_weights:
        if os.path.exists(cand):
            weights_path = cand
            break

    if not weights_path:
        print(f"CRITICAL ERROR: Weights file for Client {client_id} not found!")
        print(f"Checked paths: {candidate_weights}")
        sys.exit(1)

    # Search for metadata file
    metadata_path = None
    candidate_meta = [
        f"/app/weights/client{client_id}_metadata.json",
        f"/app/client{client_id}_metadata.json",
        f"/app/weights/client_{client_id}_metadata.json",
        f"/app/client_{client_id}_metadata.json",
        "/app/client_metadata.json",
        "/app/metadata.json"
    ]
    for cand in candidate_meta:
        if os.path.exists(cand):
            metadata_path = cand
            break

    if not metadata_path:
        print(f"CRITICAL ERROR: Metadata file for Client {client_id} not found!")
        print(f"Checked paths: {candidate_meta}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Open weights file in binary mode - ZERO TORCH DEPENDENCY
    with open(weights_path, "rb") as f:
        weight_bytes = f.read()

    byte_count = len(weight_bytes)
    sha256_hash = hashlib.sha256(weight_bytes).hexdigest()

    url = f"http://{server_ip}:{server_port}/submit"
    print(f"=== [CLIENT {client_id}] STARTING WEIGHT TRANSFER CONTAINER ===")
    print(f"Target Server : {url}")
    print(f"Weights File  : {weights_path} ({byte_count} bytes)")
    print(f"SHA256 Digest : {sha256_hash}")
    print(f"Metadata File : {metadata_path}")

    max_attempts = 5
    backoff_seconds = 10
    success = False

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[CLIENT {client_id}] Connection attempt {attempt}/{max_attempts} to {url}...")
            files = {
                "weights_file": (f"client{client_id}_weights.pt", weight_bytes, "application/octet-stream")
            }
            data = {
                "client_id": str(client_id),
                "metadata_json": json.dumps(metadata)
            }
            resp = requests.post(url, data=data, files=files, timeout=60)
            if resp.status_code == 200:
                success = True
                result = resp.json()
                print(f"[CLIENT {client_id}] SUCCESS: Sent {byte_count} bytes (SHA256: {sha256_hash[:12]}...) to {server_ip}:{server_port}. No training data transmitted.")
                print(f"[SERVER RESPONSE]: {result.get('message', 'Submission accepted')}")
                break
            else:
                print(f"[CLIENT {client_id}] Attempt {attempt} failed with status {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"[CLIENT {client_id}] Attempt {attempt} failed with network error: {e}")

        if attempt < max_attempts:
            print(f"[CLIENT {client_id}] Waiting {backoff_seconds} seconds before retry...")
            time.sleep(backoff_seconds)

    if not success:
        print(f"[CLIENT {client_id}] FAILURE: Could not connect to server at {server_ip}:{server_port} after {max_attempts} attempts.")
        sys.exit(1)

if __name__ == "__main__":
    main()
