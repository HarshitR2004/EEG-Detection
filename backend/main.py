import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from eeg_feature_extractor import EEGFeatureExtractor
from io import BytesIO
from channel_import_analyzer import ChannelImportanceAnalyzer
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Loading the model
model_path = os.path.join('backend', 'EEG_Classification.pt')
model = torch.jit.load(model_path, map_location=torch.device("cpu"))
model.eval() 

@app.get("/")
async def read_root():
    return {"message": "Welcome to the EEG Prediction API!"}

def preprocess_eeg(data):
    feature_extractor = EEGFeatureExtractor(signals=data, sampling_rate=256)
    features = feature_extractor.combine_features(normalize=True)
    return features

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith(".npy"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only .npy files are accepted.")

        # Read and load the .npy file
        file_content = await file.read()
        eeg_data = np.load(BytesIO(file_content))
        eeg_data = np.expand_dims(eeg_data, axis=0)

        # Preprocess the EEG data
        eeg_tensor = preprocess_eeg(eeg_data)
        eeg_tensor = torch.tensor(eeg_tensor).float()

        # Perform inference
        with torch.no_grad():
            outputs = model(eeg_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Convert logits to probabilities
            predicted_class = np.argmax(probs, axis=1).item()  # Get predicted class
            predicted_class = int(predicted_class)

        # Finding the top 3 most important channels
        channel_names = [f'C{i+1}' for i in range(19)]
        channel_importance_analyzer = ChannelImportanceAnalyzer(model, eeg_tensor, channel_names)
        top_channels = channel_importance_analyzer.get_top_channels(top_n=19)

        # Convert top_channels to a JSON-serializable format
        top_channels_serializable = []
        for channel, importance in top_channels:
            importance = float(importance) 
            top_channels_serializable.append([channel, importance])

        classes = {0: "Normal", 1: "Complex Partial Seizures", 2: "Electrographic Seizures", 3: "Video detected Seizures"}
        

        return JSONResponse(content={
            "predicted_class": classes[predicted_class],
            "top_channels": top_channels_serializable
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



