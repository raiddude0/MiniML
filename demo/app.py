import gradio as gr
import numpy as np

from miniml.linear_model import LinearRegression
from miniml.preprocessing.standard_scaler import StandardScaler


def train_demo_model():
    rng = np.random.default_rng(4)
    size = rng.uniform(600, 3200, 160)
    bedrooms = rng.integers(1, 6, 160)
    age = rng.uniform(0, 60, 160)

    X = np.column_stack([size, bedrooms, age])
    y = 50000 + 180 * size + 12000 * bedrooms - 900 * age
    y = y + rng.normal(scale=25000, size=160)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression(learning_rate=0.05, epochs=800)
    model.fit(X_scaled, y)
    return model, scaler


MODEL, SCALER = train_demo_model()


def predict_price(size, bedrooms, age):
    X = np.array([[size, bedrooms, age]])
    X_scaled = SCALER.transform(X)
    prediction = MODEL.predict(X_scaled)[0]
    return f"${prediction:,.0f}"


demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(600, 3200, value=1600, step=50, label="Home size"),
        gr.Slider(1, 6, value=3, step=1, label="Bedrooms"),
        gr.Slider(0, 60, value=15, step=1, label="Home age"),
    ],
    outputs=gr.Textbox(label="Predicted price"),
    title="MiniML Linear Regression Demo",
    description="A small Hugging Face Spaces demo using MiniML LinearRegression.",
)


if __name__ == "__main__":
    demo.launch()
