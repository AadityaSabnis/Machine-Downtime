from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

model = None
X_columns = None


def get_model():
    global model
    if model is None:
        if not os.path.exists("downtime_model.pkl"):
            raise HTTPException(status_code=500, detail="Model file not found. Please train the model first.")
        model = joblib.load("downtime_model.pkl")
    return model


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global X_columns
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {e}")

    if "Downtime" not in df.columns:
        raise HTTPException(status_code=400, detail="Dataset must contain a 'Downtime' column.")

    df.dropna(inplace=True)

    df['Machine_ID'] = df['Machine_ID'].astype('category').cat.codes
    df['Assembly_Line_No'] = df['Assembly_Line_No'].astype('category').cat.codes

    X_columns = [
        'Machine_ID', 'Assembly_Line_No', 'Hydraulic_Pressure(bar)',
        'Coolant_Pressure(bar)', 'Air_System_Pressure(bar)', 'Coolant_Temperature',
        'Hydraulic_Oil_Temperature(?C)', 'Spindle_Bearing_Temperature(?C)',
        'Spindle_Vibration(?m)', 'Tool_Vibration(?m)', 'Spindle_Speed(RPM)',
        'Voltage(volts)', 'Torque(Nm)', 'Cutting(kN)'
    ]

    df.to_csv("uploaded_data.csv", index=False)

    return {"message": "File uploaded successfully!"}


@app.post("/train")
def train_model():
    global model, X_columns
    if not X_columns:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a file first.")

    df = pd.read_csv("uploaded_data.csv")

    X = df[X_columns]
    y = df['Downtime'].astype('category').cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    joblib.dump(model, "downtime_model.pkl")

    return {"message": "Model trained successfully!", "accuracy": accuracy, "f1_score": f1}


class PredictInput(BaseModel):
    Machine_ID: int
    Assembly_Line_No: int
    Hydraulic_Pressure: float = Field(alias="Hydraulic_Pressure(bar)")
    Coolant_Pressure: float = Field(alias="Coolant_Pressure(bar)")
    Air_System_Pressure: float = Field(alias="Air_System_Pressure(bar)")
    Coolant_Temperature: float
    Hydraulic_Oil_Temperature_C: float = Field(alias="Hydraulic_Oil_Temperature(?C)")
    Spindle_Bearing_Temperature_C: float = Field(alias="Spindle_Bearing_Temperature(?C)")
    Spindle_Vibration_um: float = Field(alias="Spindle_Vibration(?m)")
    Tool_Vibration_um: float = Field(alias="Tool_Vibration(?m)")
    Spindle_Speed_RPM: float = Field(alias="Spindle_Speed(RPM)")
    Voltage_volts: float = Field(alias="Voltage(volts)")
    Torque_Nm: float = Field(alias="Torque(Nm)")
    Cutting_kN: float = Field(alias="Cutting(kN)")

    class Config:
        allow_population_by_field_name = True

@app.post("/predict")
def predict(input_data: PredictInput, model=Depends(get_model)):
    input_df = pd.DataFrame([input_data.dict(by_alias=True)])

    missing_cols = set(X_columns) - set(input_df.columns)
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing feature columns: {missing_cols}")

    try:
        prediction = model.predict(input_df)
        confidence = model.predict_proba(input_df).max()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    result = "Yes" if prediction[0] == 1 else "No"
    return {"Downtime": result, "Confidence": round(confidence, 2)}
