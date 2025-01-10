import os
import json
import openwakeword
from pathlib import Path
import argparse

def search_models(directory="F:\\models\\"):
    model_paths = []
    for d, _, files in os.walk(directory):
        model_paths.extend([os.path.join(d, f) for f in files if f.endswith(".onnx")])
    models_dict = {k: v for k, v in enumerate(model_paths)}
    return models_dict

def select_model(models_dict):
    print(json.dumps(models_dict, indent=4))
    selected_models = input(f"Select any models from the list: ")

# Load a trained openWakeWord models
def load_models():
    root = os.getcwd()
    model_path = os.path.join(root, "models", "alexa_ca_25000_15000_100.onnx")
    oww_alexa_model = os.path.join(root, "models", "alexa_v0.1.onnx")
    model_name = "alexa_ca_25000_15000_100"
    oww = openwakeword.Model(
        wakeword_model_paths=[model_path]
    )

# Estimate False accept rate/false positive rate
def dipco_predict():
    clips = [str(i) for i in Path("DiPCo/audio/").glob("**/*U01.CH1.wav")]
    predictions_dipco = []
    for clip in clips:
        predictions = oww.predict_clip(clip)
        predictions = [i[model_name] for i in predictions]
        predictions_dipco.append(predictions)
    len(predictions_dipco)
    return

def main():
    return

if __name__ == "main":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="The path to the model file (required)",
        type=str,
        required=True
    )
    parser.add_argument(
        "--positive",
        help="Execute the model training process",
        action="store_true",
        default="",
        required=False
    )

    args = parser.parse_args()
    main()