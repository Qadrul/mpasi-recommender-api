# convert_to_tflite.py — jalankan di lokal
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="mpasi")
class PenaltyInteractionLayer(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.penalty_weight = self.add_weight(
            name="penalty_weight", shape=(1,),
            initializer="zeros", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        score, penalty_signal = inputs
        pw = tf.sigmoid(self.penalty_weight)
        penalised = score * (1.0 - pw * penalty_signal)
        return tf.clip_by_value(penalised, 0.0, 1.0)

    def get_config(self):
        return super().get_config()

# Load model .keras
model = tf.keras.models.load_model("model/mpasi_recommender.keras")

# Convert ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantization ringan
tflite_model = converter.convert()

# Simpan
with open("model/mpasi_recommender.tflite", "wb") as f:
    f.write(tflite_model)

print("Done! Ukuran:", len(tflite_model) / 1024, "KB")