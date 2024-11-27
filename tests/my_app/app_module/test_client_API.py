import os
import pytest
from unittest.mock import patch, Mock
from my_app.app_module.client_API import (
    connector_create_predictions,
    connector_end_analysis,
    connector_abort_analysis,
)
import requests

# Common variables
ANALYSIS_ID = "test_analysis"
TOKEN = "test_token"
PAYLOAD = {"data": "test_payload"}

### Tests for `connector_create_predictions`

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.put")
def test_connector_create_predictions_success(mock_put, mock_getenv):
    mock_put.return_value = Mock(status_code=200)
    status_code, message = connector_create_predictions(ANALYSIS_ID, PAYLOAD, TOKEN)
    assert status_code == 200
    assert "Correctly sent to connector" in message

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.put", side_effect=requests.ConnectionError("Connection failed"))
def test_connector_create_predictions_connection_error(mock_put, mock_getenv):
    status_code, message = connector_create_predictions(ANALYSIS_ID, PAYLOAD, TOKEN)
    assert status_code == 500
    assert "Connection failed" in message

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.put", side_effect=requests.exceptions.InvalidSchema("Invalid schema"))
def test_connector_create_predictions_invalid_schema(mock_put, mock_getenv):
    status_code, message = connector_create_predictions(ANALYSIS_ID, PAYLOAD, TOKEN)
    assert status_code == 400
    assert "Invalid schema" in message

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.put")
def test_connector_create_predictions_http_error(mock_put, mock_getenv):
    # Set up mock to raise an HTTPError with a 403 status code
    mock_response = Mock()
    mock_response.status_code = 403
    mock_response.raise_for_status.side_effect = requests.HTTPError("Access forbidden")
    mock_put.return_value = mock_response
    
    # Call the function
    status_code, message = connector_create_predictions(ANALYSIS_ID, PAYLOAD, TOKEN)
    
    # Assert expected status code and error message
    assert status_code == 403
    assert "Access forbidden" in message

### Tests for `connector_end_analysis`

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.post")
def test_connector_end_analysis_success(mock_post, mock_getenv):
    mock_post.return_value = Mock(status_code=200)
    status_code, message = connector_end_analysis(ANALYSIS_ID, TOKEN)
    assert status_code == 200
    assert "Correctly sent to connector: end of analysis" in message

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.post", side_effect=requests.ConnectionError("Connection failed"))
def test_connector_end_analysis_connection_error(mock_post, mock_getenv):
    status_code, message = connector_end_analysis(ANALYSIS_ID, TOKEN)
    assert "Connection failed" in message

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.post", side_effect=requests.exceptions.InvalidSchema("Invalid schema"))
def test_connector_end_analysis_invalid_schema(mock_post, mock_getenv):
    status_code, message = connector_end_analysis(ANALYSIS_ID, TOKEN)
    assert "Invalid schema" in message

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.post")
def test_connector_end_analysis_http_error(mock_post, mock_getenv):
    # Set up mock to raise an HTTPError with a 404 status code
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error: Not Found for url")
    mock_post.return_value = mock_response
    
    # Call the function
    status_code, message = connector_end_analysis(ANALYSIS_ID, TOKEN)
    
    # Assert expected status code and error message
    assert status_code == 404
    assert "Resource not found" in message

### Tests for `connector_abort_analysis`

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.post")
def test_connector_abort_analysis_success(mock_post, mock_getenv):
    mock_post.return_value = Mock(status_code=200)
    status_code, message = connector_abort_analysis(ANALYSIS_ID, TOKEN)
    assert status_code == 200
    assert "Correctly sent to connector: abort analysis" in message

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.post", side_effect=requests.ConnectionError("Connection failed"))
def test_connector_abort_analysis_connection_error(mock_post, mock_getenv):
    status_code, message = connector_abort_analysis(ANALYSIS_ID, TOKEN)
    assert status_code == 500
    assert "Connection failed" in message

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.post", side_effect=requests.exceptions.InvalidSchema("Invalid schema"))
def test_connector_abort_analysis_invalid_schema(mock_post, mock_getenv):
    status_code, message = connector_abort_analysis(ANALYSIS_ID, TOKEN)
    assert "Invalid schema" in message

@patch("os.getenv", side_effect=lambda key: "/mock_address" if key == "CONNECTOR_ADDRESS" else "8080")
@patch("requests.post")
def test_connector_abort_analysis_http_error(mock_post, mock_getenv):
    # Set up mock to raise an HTTPError with a 500 status code
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error: Internal Server Error for url")
    mock_post.return_value = mock_response
    
    # Call the function
    status_code, message = connector_abort_analysis(ANALYSIS_ID, TOKEN)
    
    # Assert expected status code and error message
    assert status_code == 500
    assert "Internal server error on the connector's end" in message