# MLFLOW_TRACKING_URI = '../models/mlruns'
# MLFLOW_RUN_ID = "493ac2925d734815b116ac8b5c5f4be9"
# CLUSTERS_YAML_PATH = "../data/processed/features_skills_clusters_description.yaml"

# #------------------------------------------

# from JobPrediction import JobPrediction
# from fastapi import FastAPI

# # @asynccontextmanager
# # async def lifespan():
# #     yield
    
# app = FastAPI()

# jobPred = JobPrediction(MLFLOW_TRACKING_URI,MLFLOW_RUN_ID,CLUSTERS_YAML_PATH)

# # Create prediction endpoint 
# @app.post('/predict_jobs_probs')
# def predict_job(available_skills: list[str]):
#     predictions = jobPred.predict_jobs_probabilities(available_skills)
#     return predictions.to_dict()


from typing import List
from fastapi import FastAPI
import pandas as pd

from JobPrediction import JobPrediction

# -------------------------------------------------
# Model config
MLFLOW_TRACKING_URI = "../models/mlruns"
MLFLOW_RUN_ID = "493ac2925d734815b116ac8b5c5f4be9"
CLUSTERS_YAML_PATH = "../data/processed/features_skills_clusters_description.yaml"

# -------------------------------------------------
# FastAPI app
app = FastAPI(title="Job Prediction API")

# Load model once
job_model = JobPrediction(
    mlflow_uri=MLFLOW_TRACKING_URI,
    run_id=MLFLOW_RUN_ID,
    clusters_yaml_path=CLUSTERS_YAML_PATH
)

# -------------------------------------------------
# FastAPI endpoint
@app.post("/predict_jobs_probs")
def predict_jobs_probs(available_skills: List[str]):
    predictions = job_model.predict_jobs_probabilities(available_skills)
    return predictions.to_dict()

# # -------------------------------------------------
# # Gradio function (calls FastAPI logic directly)
# def gradio_predict(skills_text: str):
#     skills = [s.strip() for s in skills_text.split(",") if s.strip()]

#     predictions = job_model.predict_jobs_probabilities(skills)

#     df = (
#         pd.Series(predictions)
#         .sort_values(ascending=False)
#         .reset_index()
#     )
#     df.columns = ["Job", "Probability"]
#     return df

# # -------------------------------------------------
# # Gradio UI
# gradio_app = gr.Interface(
#     fn=gradio_predict,
#     inputs=gr.Textbox(
#         label="Enter skills (comma-separated)",
#         placeholder="python, sql, machine learning"
#     ),
#     outputs=gr.Dataframe(label="Predicted Job Probabilities"),
#     title="Job Prediction System",
# )

# # Mount Gradio on FastAPI
# app = gr.mount_gradio_app(app, gradio_app, path="/ui")

