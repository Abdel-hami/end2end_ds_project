import gradio as gr
import requests
import pandas as pd

API_URI = "http://127.0.0.1:5000/predict_jobs_probs"
def gradio_predict(skills_text: str):
    skills = [s.strip() for s in skills_text.split(",") if s.strip()]
    response = requests.post(
        API_URI,
        json = skills
    )
    response.raise_for_status()

    predictions = response.json()
    df = (
        pd.Series(predictions)
        .sort_values(ascending=False)
        .reset_index()
    )
    df.columns = ["Job", "Probability"]

    return df

gradio_app = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        label="Enter skills (comma-separated)",
        placeholder="python, sql, machine learning"
    ),
    outputs=gr.Dataframe(label="Predicted Job Probabilities"),
    title="Job Prediction System",
)

if __name__ == "__main__":
    gradio_app.launch()