import yaml
import os
import requests
from PIL import Image
import io
import uuid
from datetime import datetime
import csv

def save_to_csv(data, csv_file, include_headers=True):
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write headers if the file is being created and headers are needed
        if not file_exists and include_headers:
            writer.writerow(['ID', 'Description Prompt', 'Generation Prompt', 'Image URL'])

        writer.writerow(data)

    print(f"Data saved to CSV for ID: {data[0]}")


def generate_unique_id():
    # Generate a random UUID
    unique_id = uuid.uuid4()

    # Optionally, add a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a filename with UUID (and timestamp)
    filename = f"{timestamp}_{unique_id}"
    return filename


def load_config(**kwargs):
    """
    Load configuration settings from a YAML file.

    This function reads configuration settings from a specified YAML file.
    It's used to setup parameters such as data directory, file patterns, and required fields for data ingestion.

    :param config_name: The name of the configuration file (without the file extension). Defaults to 'config'.
    :type config_name: str
    :return: The configuration settings.
    :rtype: dict
    :raises Exception: If there is an issue in reading or parsing the YAML file.
    """
    
    # get parameters from kwargs
    config_file = kwargs.get('config_name','config')    
    config_filetype = kwargs.get('config_type','.yaml')
    config_relative_path = kwargs.get('config_rel_path','config')
    
    # make the path to the config file
    config_filename = config_file + config_filetype
    project_root = os.path.dirname(os.path.abspath(__file__))
    config_directory = os.path.join(project_root, config_relative_path)
    config_path = os.path.join(config_directory,config_filename)
    
    #load config file
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Failed to load config file: {e}")
        raise
    
def download_and_convert_image(image_url, image_full_path_and_name):
    try:

        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()

        # Open the image using Pillow and convert to JPG
        image = Image.open(io.BytesIO(response.content))
        rgb_image = image.convert('RGB')  # Convert to RGB in case the PNG is in RGBA format

        # Save the image as JPG
        rgb_image.save(image_full_path_and_name, format='JPEG')
        print(f"Image saved to {image_full_path_and_name}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")
        
def generate_file_location(file_path, id,file_type):
    try:
        # Combine save_path and id to create the complete file path
        file_path = os.path.join(file_path, id + file_type)
        return file_path
    except Exception as e:
        print(f"An error occurred while saving the image: {e}")
