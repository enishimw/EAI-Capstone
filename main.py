from openai import OpenAI
import requests
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
openAiClient = OpenAI(api_key='')

# Your app's API endpoint
APP_API_ENDPOINT = 'https://x.clinicplus.pro/api'

# Function to call your app's API with a specified HTTP method
def call_app_api(action, data, method='POST'):
    url = f"{APP_API_ENDPOINT}/{action}"
    
    # Handle different HTTP methods
    if method.upper() == 'GET':
        response = requests.get(url, params=data)
    elif method.upper() == 'POST':
        response = requests.post(url, json=data)
    elif method.upper() == 'PUT':
        response = requests.put(url, json=data)
    elif method.upper() == 'DELETE':
        response = requests.delete(url, json=data)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")
    
    # Return the API response
    return response.json()

# Function to process the user prompt using the LLM
def process_prompt_with_llm(prompt):
    response = openAiClient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("response", response)

    return response.choices[0].message

# Function to determine the action based on the LLM's response
def determine_action(llm_response):
    # Example logic to determine action and HTTP method
    if "create" in llm_response.lower():
        return "create", {"data": "example data"}, "POST"
    elif "update" in llm_response.lower():
        return "update", {"id": 1, "data": "updated data"}, "PUT"
    elif "delete" in llm_response.lower():
        return "delete", {"id": 1}, "DELETE"
    elif "fetch" in llm_response.lower() or "read" in llm_response.lower():
        return "read", {"id": 1}, "GET"
    else:
        return "read", {"id": 1}, "GET"  # Default to GET for unknown actions

# Flask route to handle incoming prompts from your app
@app.route('/process-prompt', methods=['POST'])
def process_prompt():
    # Get the prompt from the request
    user_prompt = request.json.get('prompt')

    print("prompted: ", user_prompt)
    
    # Process the prompt with the LLM
    llm_response = process_prompt_with_llm(user_prompt)

    print("LLM says:", llm_response)
    
    # Determine the action, data, and HTTP method based on the LLM's response
    action, data, method = determine_action(llm_response)
    
    # Call your app's API endpoint with the specified method
    api_response = call_app_api(action, data, method)
    
    # Return the API response to your app
    return jsonify(api_response)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='localhost', port=5000)