import os
import json
from typing import List, Dict
from openai import OpenAI
import requests
from flask import Flask, request, jsonify
from ChatHistoryManager import ChatHistoryManager
from LabTestAgent import LabTestAgent
from LabResultAnalysisAgent import LabResultAnalysisAgent
from langchain_core.messages import AIMessage

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
chat_history_path = "chat_histories/"
physicians_path = "physicians.json"

chat_manager = ChatHistoryManager(storage_path=chat_history_path)
lab_test_agent = LabTestAgent(
        api_key=api_key,
        auth_token=auth_token,
        consultations_path=consultations_path,
        physicians_path=physicians_path,
        lab_tests_path=lab_tests_path,
        chat_history_path=chat_history_path
    )
lab_result_agent = LabResultAnalysisAgent(api_key=api_key, lab_tests_path=lab_tests_path)

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

def save_lab_session_to_file(
        user_id: str, user_type: str, procedure_id: str, 
        operation_id: str, patient_info: Dict = None) -> str:
    """
    Save lab session data to a JSON file and return the file path.
    """
    # Generate a unique file name
    filename = f"{user_id}_{procedure_id}_lab_session.json"
    filepath = os.path.join(SESSIONS_FOLDER, filename)

    session_data = {
        "user_id": user_id,
        "user_type": user_type,
        "procedure_id": procedure_id,
        "operation_id": operation_id,
        "patient_info": patient_info or {},
        "results": {}
    }
    
    # Save session data to the file
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

    try:
        chat_id = f"{user_id}_{procedure_id}"
        chat_history = chat_manager.get_chat_history(chat_id)
    except ValueError:
        # Create new chat session if it doesn't exist
        metadata = {
            "type": "doctor_lab_request",
        }
        chat_id = chat_manager.create_chat_session(
            user_id=user_id,
            procedure_id=procedure_id,
            metadata=metadata
        )

        # Generate a welcoming message using the LLM
        welcome_prompt = f"""
        You are a friendly and helpful medical assistant. A new session has started for user {user_id}. 
        Please welcome them warmly and let them know you're here to assist with their medical needs.
        """

        # Use the LLM to generate the welcome message
        welcome_message = lab_test_agent.llm.invoke(welcome_prompt)
    
        chat_manager.add_message(
            chat_id=chat_id,
            message=AIMessage(content=welcome_message.content)
        )
    
        chat_history = chat_manager.get_chat_history(chat_id)
    
    # Format the chat history for the front-end
    formatted_chat_history = []
    for msg in chat_history:
        if isinstance(msg, AIMessage):
            formatted_chat_history.append({
                "type": msg.type,
                "content": msg.content,
                "timestamp": msg.timestamp,
            })
        else:
            formatted_chat_history.append({
                "type": msg.type,
                "content": msg.content,
                "timestamp": msg.timestamp,
            })
    
    # Return the welcome message in the response
    return jsonify({
        'status': 'success',
        'content': formatted_chat_history
    })

@app.route('/initialize-lab-session', methods=['POST'])
def initialize_lab_session():
    """Initialize a session for a lab technician"""
    # Get the data from the request
    session_data = request.json
    
    # Store the data in the session
    user_id = session_data.get('user_id')
    user_type = session_data.get('user_type')
    procedure_id = session_data.get('procedure_id')
    operation_id = session_data.get('operation_id')
    patient_info = session_data.get('patient_info', {})

    # Save session data
    save_lab_session_to_file(user_id, user_type, procedure_id, operation_id, patient_info)

    try:
        chat_id = f"{user_id}_{procedure_id}"
        chat_history = chat_manager.get_chat_history(chat_id)
    except ValueError:
        # Create new chat session if it doesn't exist
        metadata = {
            "type": "doctor_lab_request",
        }
        chat_id = chat_manager.create_chat_session(
            user_id=user_id,
            procedure_id=procedure_id,
            metadata=metadata
        )
    
        # Generate a welcoming message for the lab technician
        welcome_prompt = f"""
        You are a knowledgeable laboratory assistant. A new lab session has started for user {user_id}. 
        Please welcome them warmly and let them know you're here to help analyze and interpret lab test results.
        """

        # Use the LLM to generate the welcome message
        welcome_message = lab_result_agent.llm.invoke(welcome_prompt)

        chat_manager.add_message(
            chat_id=chat_id,
            message=AIMessage(content=welcome_message.content)
        )
    
        chat_history = chat_manager.get_chat_history(chat_id)
    
    # Format the chat history for the front-end
    formatted_chat_history = []
    for msg in chat_history:
        if isinstance(msg, AIMessage):
            formatted_chat_history.append({
                "type": msg.type,
                "content": msg.content,
                "timestamp": msg.timestamp,
            })
        else:
            formatted_chat_history.append({
                "type": msg.type,
                "content": msg.content,
                "timestamp": msg.timestamp,
            })
    
    # Return the welcome message in the response
    return jsonify({
        'status': 'success',
        'content': formatted_chat_history
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
        chat_type="doctor_lab_request",
        prompt=user_prompt,
        session_filepath=f"sessions/{user_id}_{procedure_id}_session.json"
    )
    
    # Return the result to your app
    return jsonify({
        'status': 'success',
        'content': result
    })

@app.route('/process-lab-prompt', methods=['POST'])
def process_lab_prompt():
    """Handle general prompts/questions from lab technicians"""
    # Get the prompt from the request
    user_prompt = request.json.get('prompt')
    user_id = request.json.get('user_id')
    procedure_id = request.json.get('procedure_id')

    # Process the prompt with the LabResultAnalysisAgent
    result = lab_result_agent.process_request(
        chat_type="lab_result_analysis",
        prompt=user_prompt,
        session_filepath=f"sessions/{user_id}_{procedure_id}_lab_session.json"
    )
    
    # Return the result to your app
    return jsonify({
        'status': 'success',
        'content': result
    })

@app.route('/process-lab-result', methods=['POST'])
def process_lab_result():
    """Handle individual lab test results as they're entered"""
    # Get the test data from the request
    lab_test_result = request.json.get('lab_test_result')
    user_id = request.json.get('user_id')
    procedure_id = request.json.get('procedure_id')

    prompt = f"""
    Please use the analyze_result tool to provide analysis for this lab test result:
    
    {json.dumps(lab_test_result, indent=2)}
    
    I need:
    1. A numerical score indicating if the value is above, below, or within the normal range:
       - 1: dangerously above normal range
       - 0.5: acceptably above normal range
       - 0: within normal range or not applicable
       - -0.5: acceptably below normal range
       - -1: dangerously below normal range
    2. A one-phrase insight/description of the result
    3. A detailed analysis including clinical significance
    
    Please analyze this result in the context of the patient's information.

    The response should be a valid JSON object with the fields: abnormality_score, insight, and analysis.
    Please provide your analysis in the following JSON format:
    {{
        "abnormality_score": <number>,
        "insight": <string>,
        "analysis": <string>
    }}
    """
    
    # Process the result with the LabResultAnalysisAgent
    analysis = lab_result_agent.process_result(
        chat_type="lab_result_analysis",
        prompt=prompt,
        lab_test_result=lab_test_result,
        session_filepath=f"sessions/{user_id}_{procedure_id}_lab_session.json"
    )

    # Try to parse the response as JSON
    try:
        # First, try to parse the whole response as JSON
        analysis_json = json.loads(analysis)
        
        # Return the JSON response
        return jsonify({
            'status': 'success',
            'content': analysis_json
        })
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from the text (in case it's wrapped in explanatory text)
        import re
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, analysis, re.DOTALL)
        
        if match:
            try:
                json_str = match.group(1)
                analysis_json = json.loads(json_str)
                return jsonify({
                    'status': 'success',
                    'content': analysis_json
                })
            except json.JSONDecodeError:
                pass
                
        # If all else fails, return the text as is
        return jsonify({
            'status': 'success',
            'content': {
                'abnormality_score': 0,
                'insight': 'Error parsing result',
                'analysis': analysis
            }
        })

@app.route('/get-comprehensive-analysis', methods=['POST'])
def get_comprehensive_analysis():
    """Get a comprehensive analysis of all current test results"""
    # Get the request data
    user_id = request.json.get('user_id')
    procedure_id = request.json.get('procedure_id')
    lab_tests_with_results = request.json.get('results')

    session_filepath = f"sessions/{user_id}_{procedure_id}_lab_session.json"
    # Load the current session data
    with open(session_filepath, 'r') as f:
        session_data = json.load(f)

    session_data['results'] = lab_tests_with_results
    # Save the updated session data back to the file
    with open(session_filepath, 'w') as f:
        json.dump(session_data, f, indent=2)

    lab_result_agent.set_session_context(session_data)
    
    prompt = """
        Please use the get_comprehensive_analysis tool to provide a comprehensive analysis of all current test results.
        
        Lab test results format information:
        - Each test has an o_id (unique identifier), locale_name (display name), the actual result, and unit_of_measure
        - The result field contains:
            - ready (boolean): true if the test is completed
            - status (integer): 0 = not done, 1 = standby, 2 = ready/completed
            - value (for simple tests): the actual test result value
            - option (i.e positive/negative): This is for tests where the result is chosen from a set of options
            - children (for complex tests like CBC): an array of sub-tests, each with their own results
        
        - Only analyze results with status = 2 (ready) and ready = true
        - For tests with children (like CBC), check each child test's result.ready and result.status values
        
        Please identify abnormal values (outside reference ranges), potential patterns, and clinical significance.
        Organize your analysis by test category and highlight critical values.
    """

    # Use the LabResultAnalysisAgent to get comprehensive analysis
    result = lab_result_agent.process_request(
        chat_type="lab_result_analysis",
        prompt=prompt,
        session_filepath=f"sessions/{user_id}_{procedure_id}_lab_session.json"
    )
    
    # Return the analysis to your app
    return jsonify({
        'status': 'success',
        'content': result
    })

@app.route('/submit-final-report', methods=['POST'])
def submit_final_report():
    """Generate final report and store it in the doctor's chat session"""
    print("request", request.json)
    # Get the request data
    lab_tech_comment = request.json.get('lab_tech_comment')
    user_id = request.json.get('user_id')  # Lab technician's user ID
    procedure_id = request.json.get('procedure_id')
    doctor_id = request.json.get('doctor_id')  # ID of the doctor who requested the tests
    
    # Load the lab session
    session_filepath = f"sessions/{user_id}_{procedure_id}_lab_session.json"
    
    if not os.path.exists(session_filepath):
        return jsonify({
            'status': 'error',
            'message': 'Lab session data not found'
        }), 404
    
    # Load the current session data
    with open(session_filepath, 'r') as f:
        session_data = json.load(f)
    
    # Set the session context for the agent
    lab_result_agent.set_session_context(session_data)
    
    # Generate the final report using the saved analysis and lab tech comments
    final_report = lab_result_agent.generate_final_report(lab_tech_comment)
    
    # Store the report in the doctor's chat session
    doctor_chat_id = f"{doctor_id}_{procedure_id}"
    
    try:
        # Try to add to existing chat
        chat_manager.add_message(
            chat_id=doctor_chat_id,
            message=AIMessage(content=f"## Lab Results Report\n\n{final_report}")
        )
    except ValueError:
        # If chat doesn't exist, create a new one
        chat_manager.create_chat_session(
            user_id=doctor_id,
            procedure_id=procedure_id,
            metadata={
                "type": "doctor_lab_request"
            }
        )
        chat_manager.add_message(
            chat_id=doctor_chat_id,
            message=AIMessage(content=f"## Lab Results Report\n\n{final_report}")
        )
    
    # Return success response
    return jsonify({
        'status': 'success',
        'message': 'Final report generated and stored in doctor\'s chat',
        'report': final_report
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(host='localhost', port=5000)