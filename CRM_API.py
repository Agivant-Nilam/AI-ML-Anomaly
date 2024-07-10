from flask import Flask, request, jsonify, render_template
import json
import os
from datetime import datetime

app = Flask(__name__)

# Path to the JSON file
JSON_FILE_PATH = 'data.json'

# Initialize the JSON file with an empty list if it doesn't exist
if not os.path.exists(JSON_FILE_PATH):
    with open(JSON_FILE_PATH, 'w') as json_file:
        json.dump([], json_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_data():
    data = request.json
    subject = data.get('Subject')
    body = data.get('Body')
    
    # Generate the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Load existing data from the JSON file
    with open(JSON_FILE_PATH, 'r') as json_file:
        existing_data = json.load(json_file)
    
    # Append the new data
    existing_data.append({
        'TimeStamp': timestamp,
        'Subject': subject,
        'Email_Text': body
    })

    # Save updated data back to the JSON file
    with open(JSON_FILE_PATH, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    return jsonify({"message": "Data saved successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)
