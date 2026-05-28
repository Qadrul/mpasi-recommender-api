import pickle, numpy as np, tflite_runtime.interpreter as tflite
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

interpreter = tflite.Interpreter(model_path="model/mpasi_recommender.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("model/mlbs.pkl",    "rb") as f: mlbs    = pickle.load(f)
with open("model/scalers.pkl", "rb") as f: scalers = pickle.load(f)

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

def preprocess(req):
    mk  = mlbs["makanan_kesukaan"].transform([req.makanan_kesukaan])
    pa  = mlbs["potensi_alergi"].transform([req.potensi_alergi])
    num = np.array([[req.usia_bulan, req.berat_badan, req.tinggi_badan,
                     req.lingkar_kepala, req.lingkar_lengan,
                     req.jenis_kelamin, req.status_asi, req.jumlah_gigi,
                     mk.sum(), pa.sum()]])
    num_scaled = scalers["user"].transform(num)
    return np.concatenate([mk, pa, num_scaled], axis=1).astype(np.float32)

@app.post("/recommend")
def recommend(req: RecommendRequest):
    vec = preprocess(req)
    interpreter.set_tensor(input_details[0]["index"], vec)
    interpreter.invoke()
    scores  = interpreter.get_tensor(output_details[0]["index"])[0]
    top_idx = np.argsort(scores)[::-1][:req.top_k]
    return {"recommendations": [
        {"rank": i+1, "index": int(idx), "score": round(float(scores[idx]), 4)}
        for i, idx in enumerate(top_idx)
    ]}

@app.get("/health")
def health(): return {"status": "ok"}