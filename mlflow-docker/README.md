# MLOps - MLflow + Docker + FastAPI

## Setup Instructions

1. Start MLflow & FastAPI with Docker Compose:
    ```bash
    docker-compose up --build
    ```

2. Train & Log Model:
    - Run `train_model.ipynb` in Jupyter Notebook or VSCode.

3. Test Prediction API:
    - After training, call:
    ```
    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}'
    ```

4. Access MLflow UI:
    - Open [http://localhost:5000](http://localhost:5000)

5. Access FastAPI Docs:
    - Open [http://localhost:8000/docs](http://localhost:8000/docs)

## Notes
- Model name in MLflow: **GlobalInnovationModel**
- Environment Variables:
    - `MLFLOW_TRACKING_URI` should point to the MLflow server.
