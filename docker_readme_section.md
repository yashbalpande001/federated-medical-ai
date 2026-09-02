# Docker Containerization & Deployment Guide

This guide details how to build and execute the Federated Medical AI training pipeline using containerized Docker environments.

---

## 1. Containerization Architecture

- **`Dockerfile.server`**: Builds lightweight `python:3.10-slim` image running the Flower FedProx aggregation server (`src/federated/server_fedprox.py`). Exposes gRPC port `8080`.
- **`Dockerfile.client`**: Builds lightweight `python:3.10-slim` image for virtual hospital Flower clients (`src/federated/client_fedprox.py`).
- **`docker-compose.yml`**: Orchestrates 1 central server container (`fl-server`) and 5 virtual hospital client containers (`fl-client-0` to `fl-client-4`) connected via a isolated bridge network (`fl-net`).
- **Volume Mounts**: Preprocessed tensor cache (`/outputs/cache/`), dataset partition CSVs (`/outputs/client_partitions/`), and generated evaluation reports are mounted dynamically (`./outputs:/app/outputs`). Zero raw dataset files are baked into container images.

---

## 2. Quickstart Execution Commands

### Prerequisites
- Docker Engine installed (or Docker Desktop on Windows/macOS).
- Preprocessed dataset partitions generated in `outputs/client_partitions/` (Step 9).

### Step-by-Step Commands

#### 1. Build Container Images
```bash
docker-compose build
```

#### 2. Launch Containerized FL Simulation
```bash
docker-compose up
```

#### 3. Run in Detached Mode (Background)
```bash
docker-compose up -d
```

#### 4. Monitor Container Simulation Logs
```bash
docker-compose logs -f fl-server
```

#### 5. Stop and Clean Up Containers
```bash
docker-compose down
```

---

## 3. Environment & Volume Mount Isolation
The containers communicate via the internal Docker bridge network (`fl-net`), where clients resolve `fl-server:8080` automatically. Results and model checkpoints are saved directly to the host machine's `./outputs/` directory.
