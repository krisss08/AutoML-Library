import requests
import os

from constants import (
    MLOPS_URL
)

class MlflowClient(object): 
    def __init__(
        self, 
        client_id, 
        client_name, 
        response_id = "r1", 
        correlation_id = "c1"
    ): 

        self.headers = {
        "Content-Type": "application/json",
        "x-client-id": client_id,
        "x-client-name": client_name,
        "x-request-id": response_id,
        "x-correlation-id": correlation_id,
        }
        self.MLOPS_URL = MLOPS_URL


    def create_new_experiment(self, experiment_name): 
        URL = f"{self.MLOPS_URL}/api/v2/mlops/experiments"
        experiment_data={
            "experiment_name": experiment_name
        }
        response = requests.post(URL, headers=self.headers, json=experiment_data)
        return response.json()

    def create_new_run(self, model_payload):    
        URL = f"{self.MLOPS_URL}/api/v2/mlops/models"
        response = requests.post(URL, headers=self.headers, json=model_payload)
        return response.json()

    def get_model_config(self, model_id):
        URL = f"{self.MLOPS_URL}/api/v2/mlops/models/{model_id}"
        response = requests.get(URL, headers=self.headers)
        print('getting model config...')
        return response.json()
