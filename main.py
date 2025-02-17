import os
from openai import OpenAI
import requests
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

openAiClient = OpenAI(api_key=api_key)

# Your app's API endpoint
APP_API_ENDPOINT = 'https://x.clinicplus.pro/api/'

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

# Function to process the user prompt using the LLM
def process_prompt_with_llm(prompt):
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

# Flask route to handle incoming prompts from your app
@app.route('/process-prompt', methods=['POST'])
def process_prompt():
    # Get the prompt from the request
    user_prompt = request.json.get('prompt')

    print("User prompted: ", user_prompt)
    
    # Process the prompt with the LLM
    llm_response = process_prompt_with_llm(user_prompt)

    print("LLM says: ", llm_response.content)
    
    # # Determine the action, data, and HTTP method based on the LLM's response
    # action, data, method = determine_action(llm_response)
    
    # # Call your app's API endpoint with the specified method
    # api_response = call_app_api(action, data, method)
    
    # Return the API response to your app
    return jsonify({
        'status': 'success',
        'content': llm_response.content  # Ensure this is a string or JSON-serializable object
    })

def simulate_prompt():
    prompt= "hello!!!"

    llm_response = process_prompt_with_llm(prompt)

    print("Sent response...")


# Run the Flask app
if __name__ == '__main__':
    app.run(host='localhost', port=5000)