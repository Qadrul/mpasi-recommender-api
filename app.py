import os
import pickle
import numpy as np
import tensorflow as tf
import keras
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import gradio as gr

# ── Custom layer (WAJIB didefinisikan sebelum load model) ──────────────
@keras.saving.register_keras_serializable(package="mpasi")
class PenaltyInteractionLayer(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.penalty_weight = self.add_weight(
            name="penalty_weight",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        score, penalty_signal = inputs
        pw = tf.sigmoid(self.penalty_weight)
        penalised = score * (1.0 - pw * penalty_signal)
        return tf.clip_by_value(penalised, 0.0, 1.0)

    def get_config(self):
        return super().get_config()

# ── Load artifacts ─────────────────────────────────────────────────────
MODEL_PATH = "model/mpasi_recommender.keras"
model = tf.keras.models.load_model(MODEL_PATH)

with open("model/feature_dims.pkl", "rb") as f:
    feature_dims = pickle.load(f)

with open("model/mlbs.pkl", "rb") as f:
    mlbs = pickle.load(f)

with open("model/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

# ── Preprocess ─────────────────────────────────────────────────────────
def preprocess_user(makanan_kesukaan, potensi_alergi,
                    usia_bulan, berat_badan, tinggi_badan,
                    lingkar_kepala, lingkar_lengan,
                    jenis_kelamin, status_asi, jumlah_gigi):

    mk  = mlbs["makanan_kesukaan"].transform([makanan_kesukaan])
    pa  = mlbs["potensi_alergi"].transform([potensi_alergi])

    num = np.array([[usia_bulan, berat_badan, tinggi_badan,
                     lingkar_kepala, lingkar_lengan,
                     jenis_kelamin, status_asi, jumlah_gigi,
                     mk.sum(), pa.sum()]])

    num_scaled = scalers["user"].transform(num)
    user_vec   = np.concatenate([mk, pa, num_scaled], axis=1)
    return user_vec.astype(np.float32)

# ── FastAPI ────────────────────────────────────────────────────────────
app = FastAPI()

class RecommendRequest(BaseModel):
    makanan_kesukaan: List[str] = []
    potensi_alergi:   List[str] = ["Tidak Ada"]
    usia_bulan:       float = 6
    berat_badan:      float = 7.5
    tinggi_badan:     float = 65.0
    lingkar_kepala:   float = 42.0
    lingkar_lengan:   float = 13.5
    jenis_kelamin:    int   = 1
    status_asi:       int   = 1
    jumlah_gigi:      int   = 0
    top_k:            int   = 5

@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    user_vec = preprocess_user(
        req.makanan_kesukaan, req.potensi_alergi,
        req.usia_bulan, req.berat_badan, req.tinggi_badan,
        req.lingkar_kepala, req.lingkar_lengan,
        req.jenis_kelamin, req.status_asi, req.jumlah_gigi,
    )
    scores  = model.predict(user_vec, verbose=0)[0]
    top_idx = np.argsort(scores)[::-1][:req.top_k]
    return {
        "recommendations": [
            {"rank": i + 1, "index": int(idx), "score": round(float(scores[idx]), 4)}
            for i, idx in enumerate(top_idx)
        ]
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "model": "mpasi-recommender"}

# ── Gradio UI ──────────────────────────────────────────────────────────
def predict_ui(usia_bulan, berat_badan, tinggi_badan):
    user_vec = preprocess_user(
        [], ["Tidak Ada"],
        usia_bulan, berat_badan, tinggi_badan,
        42.0, 13.5, 1, 1, 0,
    )
    scores  = model.predict(user_vec, verbose=0)[0]
    top5    = np.argsort(scores)[::-1][:5]
    return {f"Rekomendasi #{i+1} (idx {idx})": round(float(scores[idx]), 4)
            for i, idx in enumerate(top5)}

iface = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Slider(6,    24,   value=6,    label="Usia (bulan)"),
        gr.Slider(3.0,  15.0, value=7.5,  label="Berat Badan (kg)"),
        gr.Slider(50.0, 100.0,value=65.0, label="Tinggi Badan (cm)"),
    ],
    outputs=gr.Label(num_top_classes=5),
    title="MPASI Recommender",
    description="Masukkan data bayi untuk mendapatkan rekomendasi MPASI.",
)

app = gr.mount_gradio_app(app, iface, path="/")