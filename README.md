# Machine-Downtime

A FastAPI-based service to predict machine downtime using logistic regression.

## Features
- Upload CSV datasets
- Train a machine learning model
- Predict downtime with confidence scores

## Endpoints
- **POST /upload**: Upload a CSV file
- **POST /train**: Train the model
- **POST /predict**: Make predictions

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `uvicorn main:app --reload`



## Installation
1. Clone the repository:
   ```
   git clone https://github.com/AadityaSabnis/Machine-Downtime.git
   ```
2. Navigate to the project directory:
   ```
   cd project-name
   ```
3. Create a virtual environment:
   ```
   python3 -m venv .venv
   ```
4. Activate the virtual environment:
   ```
   source .venv/bin/activate  # on Linux/macOS
   .venv\Scripts\activate  # on Windows
   ```
5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Start the FastAPI server:
   ```
   uvicorn app:app --reload
   ```
2. The API will be available at `http://127.0.0.1:8000`.
3. You can access the Swagger UI for testing endpoints at `http://127.0.0.1:8000/docs`.


## Example Input for `/predict`
```json
{
    "Machine_ID": 1,
    "Assembly_Line_No": 2,
    "Hydraulic_Pressure": 50.5,
    "Coolant_Pressure": 25.3,
    "Air_System_Pressure": 30.0,
    "Coolant_Temperature": 70.0,
    "Hydraulic_Oil_Temperature_C": 60.5,
    "Spindle_Bearing_Temperature_C": 45.0,
    "Spindle_Vibration_um": 1.5,
    "Tool_Vibration_um": 2.0,
    "Spindle_Speed_RPM": 3000.0,
    "Voltage_volts": 220.0,
    "Torque_Nm": 500.0,
    "Cutting_kN": 25.0
}

