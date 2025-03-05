# 📑 MLOps Coursework

This repository contains work for the **Machine Learning Operations (MLOps)** course under the **MS Data Science program**, covering **Phase 3** of the project.

---

## 📂 Folder Structure (Phase 3 - MLflow + Docker + FastAPI)

```text
.
├── mlflow-docker                  # Contains all files related to Phase 3
│   ├── docker-compose.yaml        # Defines services (MLflow + FastAPI)
│   ├── requirements.txt           # Requirements for FastAPI service
│   ├── Dockerfile                  # Dockerfile for FastAPI service
│   ├── train_model.ipynb           # Notebook to train and log model to MLflow
│   └── fastapi_service             # Folder for FastAPI prediction service
│       └── api.py                  # FastAPI app exposing prediction endpoint
├── data                           # Data files (ignored by git)
├── src                            # Phase 1 & 2 pipeline code
├── dagster_project                 # Dagster orchestration (Phase 2)
└── README.md                       # This file (Phase 3 version)

---

## 🛠️ Phase 3 - MLflow + Docker + FastAPI

In **Phase 3**, the goal is to demonstrate:

| Component         | Description |
|------------------|------------------|
| **MLflow**       | Tracks model training runs, logs metrics, hyperparameters, and artifacts, and manages the Model Registry. |
| **FastAPI**      | Exposes a REST API `/predict` endpoint to serve predictions using the latest production model from the MLflow registry. |
| **Docker Compose** | Spins up both MLflow Tracking Server and FastAPI service together for easy deployment. |

---

## 🔗 Services Overview

| Service                       | Purpose                                   | URL |
|------------------|------------------|------------------|
| **MLflow Tracking Server** | Logs metrics, manages Model Registry | [http://localhost:5000](http://localhost:5000) |
| **FastAPI Prediction Service** | Serves predictions via REST API | [http://localhost:8000/docs](http://localhost:8000/docs) |

---

## 🚀 How to Run Everything

### Step 1: Start MLflow and FastAPI

```bash
docker-compose up --build
```

---

### Step 2: Access the Services

| Service                       | URL |
|------------------|------------------|
| **MLflow Tracking Server** | [http://localhost:5000](http://localhost:5000) |
| **FastAPI Docs (Swagger)**  | [http://localhost:8000/docs](http://localhost:8000/docs) |

---

### Step 3: Train and Log a Model to MLflow Registry

Open and run `train_model.ipynb` in the `mlflow-docker` folder. This notebook will:

- Train the machine learning model.
- Log **hyperparameters**, **metrics**, and **artifacts** to MLflow.
- Register the trained model in the **MLflow Model Registry** under the name: `GlobalInnovationModel`.

---

### Step 4: Test the Prediction Endpoint

With FastAPI running, you can test the **POST /predict** endpoint by sending a payload like this:

```json
{
    "GDP": 50000,
    "Population": 1000000,
    "R&D Expenditure": 3.5
}
```

This will return a predicted Global Innovation Score.

---
