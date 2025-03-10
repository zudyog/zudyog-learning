import streamlit as st
import pandas as pd
from io import StringIO
import requests
import json
import os

API_URL = os.getenv('API_URL', 'http://localhost:8101')


def upload_to_api(file, metadata):
    url = f"{API_URL}/index_texts/"

    # Create the files and data dictionary for the request
    files = {
        'file': (file.name, file, 'multipart/form-data')
    }

    # Convert metadata dictionary to JSON string
    data = {
        'metadata': json.dumps(metadata)
    }

    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}


# Streamlit UI
st.title("File Upload with Metadata")

# File uploader
uploaded_file = st.file_uploader("Choose a file")

# Metadata input fields (example)
st.subheader("Metadata")
key1 = st.text_input("Key 1")
key2 = st.text_input("Key 2")

if uploaded_file is not None:
    # Create metadata dictionary
    metadata = {
        "document_id": key1,
        "date_uploaded": key2
    }

    # Add a button to trigger the upload
    if st.button("Upload to API"):
        with st.spinner("Uploading..."):
            result = upload_to_api(uploaded_file, metadata)

            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Upload successful!")
                st.json(result)
