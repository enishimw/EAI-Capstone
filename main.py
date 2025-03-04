import os
import json
from typing import List, Dict
from openai import OpenAI
import requests
from flask import Flask, request, jsonify
from LabTestAgent import LabTestAgent

auth_token = os.getenv('AUTH_TOKEN')

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

openAiClient = OpenAI(api_key=api_key)

# Your app's API endpoint
APP_API_ENDPOINT = 'https://x.clinicplus.pro/api/'

SESSIONS_FOLDER = "sessions"
os.makedirs(SESSIONS_FOLDER, exist_ok=True)

consultations_path = "consultation_types.json"
lab_tests_path = "lab_tests.json"

lab_test_agent = LabTestAgent(api_key=api_key, auth_token=auth_token, consultations_path=consultations_path, lab_tests_path=lab_tests_path)

# Function to call your app's API with a specified HTTP method
def call_app_api(action, data, method='POST'):
    url = f"{APP_API_ENDPOINT}/{action}"
    url = f"{APP_API_ENDPOINT}/"
    
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

    response = openAiClient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a friendly and helpful medical assistant in the Clinic Plus Pro system. Think of yourself as that knowledgeable, approachable colleague who makes the workday better - professional but also warm and engaging. You use natural, conversational language and can throw in the occasional light-hearted comment when appropriate (though always maintaining professionalism around serious medical matters).\n\nYour personality:\n- Friendly and approachable - you're the helpful colleague everyone loves working with\n- Clear and direct, but never robotic or formal\n- Quick to understand context and match others' communication styles\n- Good at reading the room - knows when to be light and when to be serious\n- Proactive about asking questions and making helpful suggestions\n- Humble and happy to learn from the medical professionals you work with\n\nYour main duties:\n\n1. Helping with Test Requests\n- Chat naturally with doctors about what tests they need\n- Ask follow-up questions conversationally if you need more info\n- Look through patient history and suggest other relevant tests\n- Keep lab techs in the loop with clear, prioritized tasks\n- Always double-check you've got everything right before moving forward\n\n2. Making Sense of Results\n- Break down lab results in clear, natural language\n- Point out anything that needs attention\n- Explain what results might mean, while deferring to the doctor's expertise\n- Answer questions and brainstorm next steps together\n- Adapt your style to who you're talking to (doctors vs lab techs)\n\n3. Handling Critical Stuff\n- Keep an eye out for concerning values\n- Give context-aware heads up about potential issues\n- Help get the right people involved quickly when needed\n- Make sure urgent messages don't get lost\n- Follow up to ensure critical info was received\n\nYour approach:\n- Have natural conversations rather than formal exchanges\n- Use clear medical terms but explain things in plain language too\n- Ask questions like a curious colleague would\n- Be proactive about offering helpful insights\n- Know when to defer to human expertise\n- Keep things light when appropriate, but always professional\n- Double-check the important stuff\n- Speak up if something seems unclear or concerning\n\nRemember: You're here to make everyone's job easier while keeping patient care top priority. Be helpful and friendly, but always thoughtful and thorough when it comes to medical matters. Think of yourself as that reliable colleague who makes work both easier and more enjoyable."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

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

def save_session_to_file(
        user_id: str, user_type: str, procedure_id: str, 
        operation_id: str, doctor_specializations: List[Dict], 
        patient_history: List[Dict], lab_waiting_room: List[Dict]) -> str:
    """
    Save patient_history to a JSON file and return the file path.
    """
    # Generate a unique file name
    filename = f"{user_id}_{procedure_id}_session.json"
    filepath = os.path.join(SESSIONS_FOLDER, filename)

    session_data = {
        "user_id": user_id,
        "user_type": user_type,
        "procedure_id": procedure_id,
        "operation_id": operation_id,
        "doctor_specializations": doctor_specializations,
        "patient_history": patient_history,
        "lab_waiting_room": lab_waiting_room
    }
    
    # Save patient_history to the file
    with open(filepath, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    return filepath

@app.route('/initialize-session', methods=['POST'])
def initialize_session():
    # Get the data from the request
    session_data = request.json
    
    # Store the data in the session
    user_id = session_data.get('user_id')
    user_type = session_data.get('user_type')
    procedure_id = session_data.get('procedure_id')
    operation_id = session_data.get('operation_id')
    patient_history = session_data.get('patient_history', [])
    doctor_specializations = session_data.get('doctor_specializations', [])
    lab_waiting_room = session_data.get('lab_waiting_room', [])

    save_session_to_file(user_id, user_type, procedure_id, operation_id, doctor_specializations, patient_history, lab_waiting_room)
    
    # Generate a welcoming message using the LLM
    welcome_prompt = f"""
    You are a friendly and helpful medical assistant. A new session has started for user {user_id}. 
    Please welcome them warmly and let them know you're here to assist with their medical needs.
    """

    # Use the LLM to generate the welcome message
    welcome_message = lab_test_agent.llm.invoke(welcome_prompt)
    
    # Return the welcome message in the response
    return jsonify({
        'status': 'success',
        'content': welcome_message.content
    })

# Flask route to handle incoming prompts from your app
@app.route('/process-prompt', methods=['POST'])
def process_prompt():
    # Get the prompt from the request
    user_prompt = request.json.get('prompt')
    user_id = request.json.get('user_id')
    procedure_id = request.json.get('procedure_id')

    # Process the prompt with the LabTestAgent
    result = lab_test_agent.process_request(
        type="doctor_lab_request",
        prompt=user_prompt,
        session_filepath=f"sessions/{user_id}_{procedure_id}_session.json"
    )
    
    # Return the result to your app
    return jsonify({
        'status': 'success',
        'content': result  # Ensure this is a string or JSON-serializable object
    })

@app.route('/clear-session', methods=['POST'])
def clear_session():
    # Clear the session data
    session.clear()
    return jsonify({'status': 'success', 'message': 'Session cleared'})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='localhost', port=5000)