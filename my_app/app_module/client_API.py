import os
import requests
import json
import logging
from typing import Dict, Optional

# Configure logging

def connector_create_predictions(analysis_id: str, payload: Dict, token: str) -> Optional[Dict]:
    connector_address = os.getenv('CONNECTOR_ADDRESS')
    connector_port = os.getenv('CONNECTOR_PORT')
    logging.info(f"Connector address: {connector_address}:{connector_port}")
    connector_url = f"http://{connector_address}:{connector_port}/analyses/{analysis_id}/predictions"
    actual_payload = {"predictionResults": [payload]}
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    try:
        response = requests.put(connector_url, data=json.dumps(actual_payload), headers=headers)
        response.raise_for_status()

    except requests.ConnectionError:
        logging.error("Failed to connect to the connector.")
        return {"status": "failure", "info": "Connection error"}

    except requests.exceptions.InvalidSchema:
        logging.error("Invalid URL schema. Check the CONNECTOR_ADDRESS.")
        return {"status": "failure", "info": "Invalid URL schema"}

    except requests.HTTPError as e:
        if response.status_code == 403:
            logging.warning("Access forbidden. Check permissions.")
        elif response.status_code == 404:
            logging.warning("Resource not found.")
        elif response.status_code == 500:
            logging.error("Internal server error on the connector's end.")
        else:
            logging.error(f"HTTP error occurred: {e}")
        return {"status": "failure", "info": f"HTTP error: {str(e)}"}

    return response


def connector_update_analysis(analysis_id: str, status: str, finishTimestamp: str, token: str) -> Optional[Dict]:
    payload = {
        "status": status,
        "finishTimestamp": finishTimestamp
    }
    connector_address = os.getenv('CONNECTOR_ADDRESS')
    connector_port = os.getenv('CONNECTOR_PORT')
    connector_url = f"http://{connector_address}:{connector_port}/analyses/{analysis_id}"

    if connector_url is None:
        logging.error("CONNECTOR_URL environment variable is not set.")
        return {"status": "failure", "info": "CONNECTOR_URL not set"}

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    try:
        response = requests.put(connector_url, json=payload, headers=headers)
        response.raise_for_status()

    except requests.ConnectionError:
        logging.error("Failed to connect to the analysis API.")
        return {"status": "failure", "info": "Connection error"}

    except requests.exceptions.InvalidSchema:
        logging.error("Invalid URL schema. Check the CONNECTOR_ADDRESS.")
        return {"status": "failure", "info": "Invalid URL schema"}

    except requests.HTTPError as e:
        if response.status_code == 403:
            logging.warning("Access forbidden. Check permissions.")
        elif response.status_code == 404:
            logging.warning("Resource not found.")
        elif response.status_code == 500:
            logging.error("Internal server error on the connector's end.")
        else:
            logging.error(f"HTTP error occurred: {e}")
        return {"status": "failure", "info": f"HTTP error: {str(e)}"}

    return response


def connector_end_analysis(analysis_id: str, token: str) -> Optional[Dict]:
    connector_address = os.getenv('CONNECTOR_ADDRESS')
    connector_port = os.getenv('CONNECTOR_PORT')
    connector_url = f"http://{connector_address}:{connector_port}/analyses/{analysis_id}/finish"

    if connector_url is None:
        logging.error("CONNECTOR_URL environment variable is not set.")
        return {"status": "failure", "info": "CONNECTOR_URL not set"}

    headers = {
        'Authorization': f'Bearer {token}'
    }

    try:
        response = requests.post(connector_url, headers=headers)
        response.raise_for_status()

    except requests.ConnectionError:
        logging.error("Failed to connect to the analysis API.")
        return {"status": "failure", "info": "Connection error"}

    except requests.exceptions.InvalidSchema:
        logging.error("Invalid URL schema. Check the CONNECTOR_ADDRESS.")
        return {"status": "failure", "info": "Invalid URL schema"}

    except requests.HTTPError as e:
        if response.status_code == 403:
            logging.warning("Access forbidden. Check permissions.")
        elif response.status_code == 404:
            logging.warning("Resource not found.")
        elif response.status_code == 500:
            logging.error("Internal server error on the connector's end.")
        else:
            logging.error(f"HTTP error occurred: {e}")
        return {"status": "failure", "info": f"HTTP error: {str(e)}"}

    return response