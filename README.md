# 📑 MLOps Coursework

This repository contains work for the **Machine Learning Operations (MLOps)** course under the **MS Data Science program**, covering **Phase 1 and Phase 2** of the project.

---

## 📂 Folder Structure

```text
.
├── data                     # Data files (ignored by git)
├── features                  # Feature engineering scripts
├── models                    # Model training, evaluation, and prediction scripts
├── src                       # Main source folder for the ML pipeline
│   ├── data                  # Data loading and preprocessing
│   ├── features               # Feature engineering module
│   ├── models                  # Model training and prediction
│   ├── utils                   # Configs and logging
│   ├── main.py                 # Pipeline entry point
├── dagster_project            # Dagster pipeline implementation
│   ├── assets                  # Dagster assets for each step
│   ├── definitions.py          # Dagster repository definitions
│   ├── io_manager               # Custom IO manager (optional)
│   ├── jobs                     # Dagster jobs
│   ├── tests                    # Dagster tests
│   └── dagster.yaml             # Dagster configuration file
└── requirements.txt            # Required Python packages

---

## 🛠️ Phase 1 - Modular ML Pipeline

In **Phase 1**, a Jupyter Notebook-based machine learning model was refactored into a fully modular Python project.

### 📂 Key Components

| Step                | Description                                   |
|--------------------|----------------------------------|
| **Data Loading**   | Reads data from `data.csv` |
| **Preprocessing**  | Handles missing values, encoding, etc. |
| **Feature Engineering** | Adds any computed or derived features |
| **Model Training** | Fits a simple `RandomForestRegressor` |
| **Prediction**      | Generates predictions on new data |
| **Evaluation**     | Calculates `MSE` and `R²` score |

---

## 🚀 Phase 2 - Dagster Orchestration

In **Phase 2**, the modular pipeline was converted into **Dagster assets** and a complete **Dagster job** was defined for orchestration.

### 📊 Dagster Features Used

| Feature            | Description |
|--------------------|------------------|
| **@asset**         | Each pipeline step became an asset |
| **Definitions**    | Defines the collection of assets for Dagster to load |
| **Dagster UI**     | Provides visibility into runs, lineage, and debugging |

---

## 🚀 Running the Pipeline

| Phase                  | Command                                                                          | Notes |
|---------------------|-----------------------------------------------------------------------------------|---|
| **Phase 1 (Manual Pipeline)** | `python -m src.main`                     | Executes directly from Python |
| **Phase 2 (Dagster Pipeline)** | `dagster dev -f dagster_project/dagster_project/definitions.py` | Starts Dagster UI & services |

- After starting Dagster, visit: [http://localhost:3000](http://localhost:3000)

---

## 💡 Key Learnings

- **Modularization** makes pipelines reusable, testable, and production-ready.
- **Dagster** provides orchestration, monitoring, and lineage tracking.
- **Pre-commit hooks** enforce code quality (linting, trailing spaces, notebook output stripping) before allowing commits.

---

## 📝 Important Notes

- `data.csv` is intentionally **ignored via `.gitignore`** to avoid pushing data to the repository.
- **Pre-commit hooks** automatically:
    - Strip outputs from Jupyter Notebooks.
    - Fix trailing spaces and missing newlines.
    - Lint using `ruff` for fast Python code analysis.

---

## 🔗 Next Phase

Phase 3 (`mlflow-docker` branch) implements:

- **Model tracking** with **MLflow**.
- **Serving predictions** with **FastAPI**.
- **Containerization** using **Docker Compose**.

---

## 📄 References

- [Dagster Documentation](https://docs.dagster.io/)


---
