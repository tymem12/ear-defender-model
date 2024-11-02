import os
import requests
import json
import logging
from typing import Dict, List, Optional


def connector_create_predictions(payload: Dict, token: str) -> Optional[Dict]:
    connector_url = os.getenv('CONNECTOR_ADDRESS') + ':' + os.getenv('CONNECTOR_PORT') + '/predictions'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    try:
        response = requests.post(connector_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    except (requests.ConnectionError, requests.exceptions.InvalidSchema) as e:
        logging.info("Failed to connect to the predictions API. Ensure the connector service is running.")
        return {
            "status": "failure",
            "info": f"Failed to connect to the predictions API. Error: {str(e)}"
        }
    except requests.HTTPError as e:
        logging.info(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
        return {
            "status": "failure",
            "info": f"HTTP error occurred: {str(e)}"
        }

    return response.json()

def connector_update_analysis(analysis_id: str, status: str, finishTimestamp: str, predictionResults: List[Dict], token:str) -> Optional[Dict]:
    # Prepare the payload for the request
    payload = {
        "status": status,
        "finishTimestamp": finishTimestamp,
        "predictionResults": predictionResults
    }

    # Get the URL from environment variables
    connector_url = os.getenv('CONNECTOR_ADDRESS') + ':' + os.getenv('CONNECTOR_PORT') + '/analyses/' + str(analysis_id) + '/'

    if connector_url is None:
        return {
            "status": "failure",
            "info": "CONNECTOR_URL environment variable is not set."
        }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    try:
        response = requests.put(connector_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses
    except (requests.ConnectionError, requests.exceptions.InvalidSchema) as e:
        logging.info("Failed to connect to the analysis API. Ensure the connector service is running.")
        return {
            "status": "failure",
            "info": f"Failed to connect to the analysis API. Error: {str(e)}"
        }
    except requests.HTTPError as e:
        logging.info(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
        return {
            "status": "failure",
            "info": f"HTTP error occurred: {str(e)}"
        }

    return response.json()
