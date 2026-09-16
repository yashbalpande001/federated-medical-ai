# 🖥️ SERVER OWNER SETUP GUIDE (`docs/SERVER_SETUP.md`)

This guide is for the **Server Owner** hosting the central aggregation node on their PC over Tailscale.

---

## 📌 Prerequisites

1. **Docker Desktop**: Installed and running.
2. **Tailscale**: Installed, connected to your mesh network, and assigned a static Tailscale IP (e.g., `100.x.y.z`).
3. **Pre-Extracted Test Set File**: `data/test_set_900.pt` (generated via `export_test_set.ipynb`).

---

## 📋 STEP-BY-STEP SERVER EXECUTION

### Step 1: Export Test Set (`data/test_set_900.pt`)
1. Run `kaggle_notebooks/export_test_set.ipynb` on Kaggle GPU.
2. Download `test_set_900.pt` from notebook outputs.
3. Place it in your local repository at:
   ```text
   federated-medical-ai/data/test_set_900.pt
   ```

> [!IMPORTANT]
> The server **will fail to start** if `data/test_set_900.pt` is missing.

---

### Step 2: Configure Environment Variables

Create or update your `.env` file in the project root:
```env
SERVER_TAILSCALE_IP=YOUR_ACTUAL_TAILSCALE_IP
SERVER_PORT=8080
```

---

### Step 3: Build & Start Server Container

Open terminal / PowerShell in `federated-medical-ai/`:

```bash
# Build Server Container
docker build -t fed-medical-server -f docker/server/Dockerfile docker/server/

# Run Server Container (Mounting data/ and outputs/)
docker run --rm -p 8080:8080 -v "${PWD}/data:/app/data:ro" -v "${PWD}/outputs:/app/outputs" --name server_aggregator fed-medical-server
```

---

### Step 4: Verify Server Health

Open a browser or terminal and test health endpoint:
```bash
curl http://localhost:8080/health
```

*Expected Output*:
```json
{"status":"online","expected_clients":2,"received_count":0,"clients_received":[]}
```

---

### Step 5: Monitor Submissions & Aggregation Results

As clients send weights over Tailscale:
1. Server logs incoming IP, client_id, sample count, byte count, and **SHA256 digest**.
2. If two clients submit identical SHA256 hashes, the server will **ABORT** immediately.
3. When `2/2` unique submissions arrive, the server runs weighted FedAvg aggregation and evaluates against `data/test_set_900.pt`.
4. The output is written to `outputs/round1_results.json` containing `"rounds_executed": 1`.
