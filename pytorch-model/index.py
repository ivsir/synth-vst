import os
import json
# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Construct the relative path to the JSON file
relative_path = '../client/chords/chords.json'  # Relative to the script's directory
file_path = os.path.join(current_dir, relative_path)

# Load the JSON file
try:
    with open(file_path, 'r') as f:
        chords_data = json.load(f)
    print("File loaded successfully.")
    
    # Print the structure of the loaded data to verify
    print("Loaded data structure:")
    print(json.dumps(chords_data, indent=2))
    
except FileNotFoundError:
    print(f"File not found: {file_path}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
