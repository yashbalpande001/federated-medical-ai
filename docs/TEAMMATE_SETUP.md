# 🤝 TEAMMATE CLIENT SETUP GUIDE (`docs/TEAMMATE_SETUP.md`)

Welcome! This simple, step-by-step guide is written for teammates participating in live Federated Learning model weight submissions. 

> [!NOTE]
> **No Python programming knowledge is required.** All commands are ready to copy and paste!

---

## 📌 1. Prerequisites (What You Need Installed)

Before starting, make sure you have the following 3 tools ready on your computer:

1. **Docker Desktop**: Download and install from [docker.com](https://www.docker.com/products/docker-desktop/). Make sure Docker Desktop is opened and running in your taskbar.
2. **Tailscale**: Download and install from [tailscale.com](https://tailscale.com/). Connect to our shared Tailscale network so your computer can reach the server.
3. **Kaggle Account**: A free account on [kaggle.com](https://www.kaggle.com/) to fine-tune your client model weights on cloud GPUs.

---

## 📋 2. STEP-BY-STEP EXECUTION GUIDE

### Step 1: Clone the Project Repository
Open PowerShell (Windows) or Terminal (Mac/Linux) and run:
```bash
git clone https://github.com/yashbalpande001/federated-medical-ai.git
cd federated-medical-ai
```

---

### Step 2: Train Your Assigned Client Weights on Kaggle GPU

1. Log into [Kaggle](https://www.kaggle.com/) and create a **New Notebook**.
2. Click **File** $\rightarrow$ **Upload Notebook**.
3. Select your assigned notebook from the repository:
   - **Client 1**: Upload `kaggle_notebooks/kaggle_client1_train.ipynb`
   - **Client 2**: Upload `kaggle_notebooks/kaggle_client2_train.ipynb`
4. In the right-hand panel under **Settings**:
   - Change **Accelerator** to **GPU T4 x2**.
5. Under **+ Add Input** (top right):
   - Search `rsna-pneumonia-detection-challenge` $\rightarrow$ Click **Add**.
   - Search your saved dataset (or `federated-medical-ai-outputs`) $\rightarrow$ Click **Add**.
6. Click **Run All** (top menu).
7. When completed, scroll to the **Output** section at the bottom and download:
   - `client1_weights.pt` and `client1_metadata.json` (if Client 1)
   - `client2_weights.pt` and `client2_metadata.json` (if Client 2)
8. Save both downloaded files into your local `federated-medical-ai/weights/` folder.

---

### Step 3: Build the Transfer Container

Open PowerShell or Terminal inside `federated-medical-ai/`:

```bash
docker build -t fed-client -f docker/client/Dockerfile docker/client/
```

---

### Step 4: Transmit Weights to the Aggregator Server

Set the server's Tailscale IP address (provided by the server owner) and run the transmission container:

#### **If you are Client 1**:
```bash
# Set Server Tailscale IP (Replace 100.x.y.z with real IP)
$env:SERVER_TAILSCALE_IP="100.x.y.z"
$env:CLIENT_ID="1"

# Run Container
docker run --rm -e SERVER_TAILSCALE_IP=$env:SERVER_TAILSCALE_IP -e CLIENT_ID=1 -v "${PWD}/weights:/app/weights:ro" fed-client
```

#### **If you are Client 2**:
```bash
# Set Server Tailscale IP (Replace 100.x.y.z with real IP)
$env:SERVER_TAILSCALE_IP="100.x.y.z"
$env:CLIENT_ID="2"

# Run Container
docker run --rm -e SERVER_TAILSCALE_IP=$env:SERVER_TAILSCALE_IP -e CLIENT_ID=2 -v "${PWD}/weights:/app/weights:ro" fed-client
```

---

### Step 5: Verify Successful Transmission

Look at the terminal output. You are looking for this **exact success log**:

```text
[CLIENT 1] Connection attempt 1/5 to http://100.x.y.z:8080/submit...
[CLIENT 1] SUCCESS: Sent 44751234 bytes (SHA256: a1b2c3d4e5f6...) to 100.x.y.z:8080. No training data transmitted.
[SERVER RESPONSE]: Client 1 submission received.
```

---

## 🛠️ 3. TROUBLESHOOTING & COMMON ISSUES

### ❌ Error: `CRITICAL ERROR: SERVER_TAILSCALE_IP environment variable is unset`
- **Cause**: You forgot to set the `SERVER_TAILSCALE_IP` variable before running `docker run`.
- **Fix**: Run `$env:SERVER_TAILSCALE_IP="100.x.y.z"` first (replace `100.x.y.z` with the server owner's real Tailscale IP).

### ❌ Error: `Attempt 1 failed with network error / Connection refused`
- **Cause**: Tailscale is disconnected or the server container is not running yet.
- **Fix 1**: Open Tailscale app and verify you are connected to the mesh network.
- **Fix 2**: Test server connection: `ping 100.x.y.z` (or ask server owner if server container is running).

### ❌ Error: `CRITICAL ERROR: Weights file for Client N not found!`
- **Cause**: You placed `client1_weights.pt` or `client1_metadata.json` in the wrong folder.
- **Fix**: Verify files are located inside `federated-medical-ai/weights/client1_weights.pt` and `client1_metadata.json`.
