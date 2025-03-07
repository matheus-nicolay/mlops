import os
import json
import mlflow
import uvicorn
import numpy as np
from pydantic import BaseModel
from fastapi import FASTAPI

class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float

app = FASTAPI(title="Fetal Health API", 
              openapi_tags=[
                  {
                    "name": "Health",
                    "description": "Get API health"
                  },
                  {
                      "name": "Prediction",
                      "description": "Model prediction"
                  }
              ])

def load_model():
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'default_url')
    mlflow_tracking_username = os.getenv('MLFLOW_TRACKING_USERNAME', 'default_user')
    mlflow_tracking_password = os.getenv('MLFLOW_TRACKING_PASSWORD', 'default_pass')

    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_tracking_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_tracking_password

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)
    registered_model = client.get_registered_model('fetal_health')
    run_id = registered_model.latest_versions[-1].run_id

    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return loaded_model

@app.on_event(event_type='startup')
def startup_event():
    global loaded_model
    loaded_model = load_model()

@app.get(path='/', 
         tags=['Health'])
def api_health():
    return {"status": "healthy"}

@app.post(path='/predict', 
         tags=['Prediction'])
def predict(request : FetalHealthData):
    global loaded_model

    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)

    prediction = loaded_model.predict(received_data)
    print(prediction)

    return {"prediction": prediction}
