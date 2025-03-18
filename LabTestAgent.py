import os
from typing import List, Dict
from langchain.tools import tool, StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
import json
import requests
from datetime import datetime

from ChatHistoryManager import ChatHistoryManager

class LabTestAgent:
    def __init__(self, api_key: str, auth_token: str, consultations_path: str, physicians_path: str, lab_tests_path: str, chat_history_path: str = "chat_histories/"):
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4o"
        )

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.auth_token = auth_token
        
        # Load and process consultations
        self.consultations = self._load_and_process_consultations(consultations_path)
        # Load and process lab tests
        self.lab_tests = self._load_and_process_tests(lab_tests_path)
        # Load and process physicians and their specializations
        self.physicians = self._load_and_process_physicians(physicians_path)

        # Initialize session context to hold global session data
        self.session_context = None
        
        self.tools = None
        self.agent = None
        self.chat_manager = ChatHistoryManager(storage_path=chat_history_path)
    
    def _load_and_process_consultations(self, consultations_path: str) -> Dict:
        """
        Load consultations and create a vector store for semantic search
        """
        # Load lab tests from JSON file
        with open(consultations_path, 'r') as f:
            consultations = json.load(f)
        
        # Prepare documents for vector store
        texts = []
        metadatas = []
        for consultation in consultations:
            texts.append(consultation['locale_name'])
            metadatas.append({'id': consultation['id']})
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        return consultations
    
    def _load_and_process_tests(self, lab_tests_path: str) -> Dict:
        """
        Load lab tests and create a vector store for semantic search
        """
        # Load lab tests from JSON file
        with open(lab_tests_path, 'r') as f:
            lab_tests = json.load(f)
        
        # Prepare documents for vector store
        texts = []
        metadatas = []
        for test in lab_tests:
            texts.append(test['locale_name'])
            metadatas.append({'o_id': test['o_id']})
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        return lab_tests
    
    def _load_and_process_physicians(self, physicians_path: str) -> Dict:
        """
        Load physicians data and create a vector store for semantic search.
        
        Args:
            physicians_path (str): Path to the physicians JSON file
            
        Returns:
            Dict: Dictionary of loaded physician data
        """
        # Load physicians data from JSON file
        with open(physicians_path, 'r') as f:
            physicians = json.load(f)
        
        # Prepare documents for vector store
        texts = []
        metadatas = []
        
        for physician in physicians:
            # Combine all specializations into a single text for better semantic matching
            specialization_texts = [spec['locale_name'] for spec in physician['specializations']]
            combined_text = f"Physician {physician['sn']} specializing in: {', '.join(specialization_texts)}"
            
            texts.append(combined_text)
            metadatas.append({'sn': physician['sn'], 'specializations': [spec['o_id'] for spec in physician['specializations']]})
        
        # Create vector store for physicians
        self.physician_vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        return physicians

    def load_session_from_file(self, filepath: str) -> Dict:
        """
        Load user_procedure session from a JSON file.
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    def _create_tools(self) -> List[Tool]:
        @tool
        def identify_consultation_type(text: str) -> str:
            """
            Identifies the type of consultation based on the doctor's specializations.
            Returns the selected consultation type.
            """
            # Get current date info
            current_date = datetime.now()
            is_weekend = current_date.weekday() >= 5  # 5=Saturday, 6=Sunday
            
            prompt = f"""Based on the doctor's specializations: "{json.dumps(self.session_context['doctor_specializations'], indent=2)}"

            Task: Identify the most appropriate consultation type from the available options:
            {json.dumps(self.consultations, indent=2)}

            Current date: {current_date.strftime("%Y-%m-%d")}
            Is weekend/holiday: {"Yes" if is_weekend else "No"}

            Process:
            1. Extract any mention of consultation type (e.g., "new consultation", "follow-up", etc.)
            2. Consider the doctor's specializations and match to relevant consultation types
            3. If today is a weekend/holiday, prefer consultation types containing "weekend, holiday"
            4. Identify the most appropriate consultation type ID from the provided list
            5. Respond with a short recommendation including the consultation type ID (as provided), the name of the consultation and a short description of the reason for the suggestion

            Example response:
            "I recommend 'New consultation - General practitioner (weekend, holiday)' (ID: 123) based on your specializations andbecause today is a weekend."
            
            Remember to be concise but clear.
            """
            response = self.llm.invoke(prompt)
            return response
        
        @tool
        def identify_lab_tests(text: str) -> List[str]:
            """
            Identifies requested lab tests using medical knowledge to match against available tests.
            """
            prompt = f"""Based on the doctor's prompt: "{text}"

            Task: Identify the specific lab tests needed from our system:
            {json.dumps(self.lab_tests, indent=2)}

            Process:
            1. Extract all mentions of specific lab tests or test categories from the prompt
            2. Match the doctor's prompt against available tests considering:
            - Standard medical terminology and common abbreviations
            - Parent-child relationships between tests
            - Complementary tests that are typically ordered together
            
            3. For each identified test, include:
            - Test name
            - o_id (required for ordering)
            - Any relevant notes about the test
            
            Example response:
            "I've identified these tests:
            1. Complete Blood Count (CBC) (o_id: 456)
            2. Comprehensive Metabolic Panel (o_id: 789)"

            Be concise but thorough in your analysis.
            """
            response = self.llm.invoke(prompt)

            return response
            
        @tool
        def analyze_patient_history() -> List[str]:
            """
            Analyzes patient history and suggests additional relevant lab tests.
            Returns a list of suggested tests with o_ids and reasons.
            """
            patient_history = self.session_context.get('patient_history', {})
            # First, check if patient_history is already a dict
            if isinstance(patient_history, dict):
                # Already parsed, no need to do anything
                pass
            elif isinstance(patient_history, str):
                try:
                    # Try to parse the input if it's a string representation of JSON
                    patient_history = json.loads(patient_history)
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, it might be a Python string representation with single quotes
                    try:
                        import ast
                        # Use ast.literal_eval which can safely evaluate Python literals
                        patient_history = ast.literal_eval(patient_history)
                    except Exception as e:
                        # If that also fails, report the error
                        return f"Error parsing patient history: Could not convert to dictionary. {str(e)}"
                except Exception as e:
                    # Catch any other unexpected errors
                    return f"Unexpected error processing patient history: {str(e)}"

            prompt = f"""Analyze this patient history and suggest additional lab tests:
            {json.dumps(patient_history, indent=2)}

            Available tests:
            {json.dumps(self.lab_tests, indent=2)}

            Consider:
            1. Previous diagnoses (e.g., Helicobacter pylori infection)
            2. Abnormal lab results (e.g., elevated eosinophils)
            3. Chronic symptoms (e.g., abdominal pain)
            4. Medication history
            5. Relevant follow-up tests

            For each suggestion:
            - Provide the test's official name
            - Include the o_id from available tests
            - Give a brief medical justification
            - Prioritize tests not already in the history

            Format response as:
            1. [Test Name] (o_id: [ID]) - [Justification]
            2. [...]

            Important:
            - Always provide justification for each suggested test.
            """

            response = self.llm.invoke(prompt).content
            return response

        @tool
        def determine_request_priority(doctor_intent: str) -> Dict:
            """
            Analyzes the doctor's intent and lab waiting room status to determine the priority score.
            The priority score ranges from -100 to 100, with 0 being the default neutral score.
            
            Args:
                doctor_intent: The doctor's message indicating urgency/priority
                lab_waiting_room: List of pending lab tests and their statuses
                
            Returns:
                Dict with priority_score and justification
            """
            prompt = f"""
            Analyze the doctor's message and lab waiting room status to determine a priority score.
            
            Doctor's message: "{doctor_intent}"
            Current lab load: "{json.dumps(self.session_context.get('lab_waiting_room', []), indent=2)}"
            
            Task: Calculate a priority score from -100 to 100 where:
            - 0 is the default neutral priority
            - 1 to 100 indicates increasing urgency (100 being highest emergency priority)
            - -1 to -100 indicates decreasing urgency (-100 being lowest priority/can wait)
            
            Consider these factors:
            1. Explicit urgency keywords (STAT, urgent, emergency, ASAP, critical, etc.)
            2. Medical conditions suggesting urgency (cardiac symptoms, severe pain, etc.)
            3. Current lab load
            4. Time-sensitivity of tests requested
            5. Contextual clues about patient condition
            
            Guidelines:
            - Scores 50-100: Reserved for life-threatening situations and emergencies
            - Scores 20-49: Urgent but not immediately life-threatening
            - Scores 1-19: Higher than standard priority
            - Score 0: Standard priority
            - Scores -1 to -30: Can wait, routine screenings
            - Scores -31 to -100: Very low priority, can be significantly delayed
            
            Output format:
            {{
                "priority_score": [integer between -100 and 100],
                "justification": [brief explanation of the score]
            }}
            """
            
            response = self.llm.invoke(prompt)
            
            # Parse the JSON response
            try:
                # First try to extract JSON if it's embedded in text
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    # If no JSON pattern found, try to parse the whole response
                    result = json.loads(response.content)
                    
                # Ensure we have the required fields
                if "priority_score" not in result or "justification" not in result:
                    result = {
                        "priority_score": 0,
                        "justification": "Could not determine priority from input. Using default priority."
                    }
                    
                # Clamp the score to the allowed range
                result["priority_score"] = max(-100, min(100, result["priority_score"]))
                    
            except Exception as e:
                # Fallback to default if parsing fails
                result = {
                    "priority_score": 0,
                    "justification": f"Error parsing priority: {str(e)}. Using default priority."
                }
            
            return result

        def submit_lab_request(operation_id: str, user_id: str, test_ids: List[str], consultation_type: str, priority_score: int = 0) -> Dict:
            """
            Submits the finalized lab test request to the external API with priority score.
            
            Args:
                operation_id: ID of the operation
                user_id: Doctor's user ID
                test_ids: List of lab test o_ids to order
                consultation_type: Type of consultation (ID)
                priority_score: Priority score from -100 to 100, with 0 as default
                
            Returns:
                Dict with success status and operation ID
            """
            api_endpoint = "https://x.clinicplus.pro/api/llm/consultations/lab_request"
            
            # Prepare the payload with priority score
            payload = {
                "operation": operation_id,
                "doctor": user_id,
                "consultation": consultation_type,
                "exams": {"exams": test_ids},
                "priority_score": priority_score
            }
            
            # Prepare the headers
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            }
            
            # Make the API request
            response = requests.post(
                api_endpoint,
                json=payload,
                headers=headers
            )

            data = json.loads(response.text)

            return {"success": data["success"], "operation_id": data["operation"]["id"]}
        
        def escalate_procedure(procedure_id: str, doctor_id: str) -> Dict:
            """
            Submits a request to escalate a procedure to another doctor/specialist.
            
            Args:
                procedure_id: The ID of the procedure to escalate
                doctor_id: The ID of the doctor/specialist to escalate to
                
            Returns:
                Dict with success status and response information
            """
            api_endpoint = "https://x.clinicplus.pro/api/llm/procedure/escalate"
            
            # Prepare the payload
            payload = {
                "procedure": procedure_id,
                "doctor": doctor_id
            }
            
            # Prepare the headers
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            }
            
            try:
                # Make the API request
                response = requests.post(
                    api_endpoint,
                    json=payload,
                    headers=headers
                )
                print(response.text)
                
                # Parse the response
                data = json.loads(response.text)

                # Get the current doctor's ID from the session context
                current_doctor_id = self.session_context.get('user_id')

                # Check if the escalation was successful
                if data.get("success", False):
                    operation_id = data.get("operation", {}).get("id")
                    
                    # Now handle post-escalation tasks
                    # 1. Load doctor A's session data
                    doctor_a_session_filepath = f"sessions/{current_doctor_id}_{procedure_id}_session.json"
                    
                    if not os.path.exists(doctor_a_session_filepath):
                        return {
                            "success": True,
                            "operation_id": operation_id,
                            "post_escalation": {
                                "success": False,
                                "message": f"Doctor A's session data not found at {doctor_a_session_filepath}"
                            }
                        }
                        
                    with open(doctor_a_session_filepath, 'r') as f:
                        doctor_a_session = json.load(f)
                    
                    # 2. Get doctor B's specializations
                    doctor_b_specializations = []
                    for physician in self.physicians:
                        if physician['sn'] == doctor_id:
                            doctor_b_specializations = physician['specializations']
                            break
                    
                    # 3. Create a new session file for doctor B
                    doctor_b_session = {
                        "user_id": doctor_id,
                        "user_type": "doctor",  # Assuming both are doctors
                        "procedure_id": procedure_id,
                        "operation_id": operation_id,
                        "doctor_specializations": doctor_b_specializations,
                        "patient_history": doctor_a_session.get("patient_history", []),
                        "lab_waiting_room": doctor_a_session.get("lab_waiting_room", []),
                        "escalated_from": {
                            "doctor_id": current_doctor_id,
                        }
                    }
                    
                    # Save doctor B's session
                    doctor_b_session_filepath = f"sessions/{doctor_id}_{procedure_id}_session.json"
                    with open(doctor_b_session_filepath, 'w') as f:
                        json.dump(doctor_b_session, f, indent=2)
                    
                    # 4. Generate a comprehensive report for doctor B
                    # Get doctor A's specialization names for the report
                    doctor_a_specialization_names = []
                    for spec in doctor_a_session.get("doctor_specializations", []):
                        doctor_a_specialization_names.append(spec.get("locale_name", "Unknown specialization"))
                    
                    # Get doctor B's specialization names for the report
                    doctor_b_specialization_names = []
                    for spec in doctor_b_specializations:
                        doctor_b_specialization_names.append(spec.get("locale_name", "Unknown specialization"))
                    
                    # Format patient history for the report
                    patient_history = doctor_a_session.get("patient_history", [])
                    
                    # Generate the report using LLM
                    report_prompt = f"""
                    You are preparing a detailed medical report for a specialist doctor who is receiving an escalated procedure.
                    
                    Escalation Details:
                    - Procedure ID: {procedure_id}
                    - Referring Doctor: {current_doctor_id} (Specializations: {', '.join(doctor_a_specialization_names)})
                    - Receiving Specialist: {doctor_id} (Specializations: {', '.join(doctor_b_specialization_names)})
                    
                    Patient History:
                    {json.dumps(patient_history, indent=2)}
                    
                    Please create a comprehensive handover report that includes:
                    1. A professional greeting and introduction
                    2. Brief summary of the patient's case and why it was escalated
                    3. Key medical findings and observations from the referring doctor
                    4. Relevant patient history details
                    5. Any lab results or tests already performed
                    6. Specific areas of concern that require the specialist's expertise
                    7. Any urgent considerations or time-sensitive issues
                    
                    Format the report professionally with clear sections and emphasis on critical information.
                    """
                    
                    report_response = self.llm.invoke(report_prompt)
                    escalation_report = report_response.content
                    
                    # 5. Create or use existing chat session for doctor B
                    doctor_b_chat_id = f"{doctor_id}_{procedure_id}"
                    
                    try:
                        # Try to get existing chat history
                        self.chat_manager.get_chat_history(doctor_b_chat_id)
                    except ValueError:
                        # Create new chat session if it doesn't exist
                        metadata = {
                            "type": "doctor_lab_request",
                            "escalated_from": current_doctor_id
                        }
                        self.chat_manager.create_chat_session(
                            user_id=doctor_id,
                            procedure_id=procedure_id,
                            metadata=metadata
                        )
                        
                        # Generate a welcome message for doctor B
                        welcome_prompt = f"""
                        You are a helpful medical assistant. Doctor {doctor_id} has received a case escalated from Doctor {current_doctor_id}.
                        Please generate a brief welcome message informing them of the escalation and that you're ready to assist.
                        Keep it professional and concise.
                        """
                        
                        welcome_message = self.llm.invoke(welcome_prompt)
                        
                        # Add welcome message to chat
                        self.chat_manager.add_message(
                            chat_id=doctor_b_chat_id,
                            message=AIMessage(content=welcome_message.content)
                        )
                    
                    # 6. Add the escalation report to doctor B's chat
                    self.chat_manager.add_message(
                        chat_id=doctor_b_chat_id,
                        message=AIMessage(content=f"## Escalation Report\n\n{escalation_report}")
                    )
                    
                    # Return combined results
                    return {
                        "success": True,
                        "operation_id": operation_id,
                        "post_escalation": {
                            "success": True,
                            "message": "Successfully created session and report for specialist doctor",
                            "report": "Report generated and added to specialist's chat session"
                        }
                    }
                else:
                    # Return the API response if escalation failed
                    return {
                        "success": data.get("success", False),
                        "message": data.get("message", "Unknown error occurred during escalation"),
                        "operation_id": None
                    }
            except Exception as e:
                # Handle any exceptions
                return {
                    "success": False,
                    "message": f"Error during escalation: {str(e)}",
                    "operation_id": None
                }

        return [
            Tool(
                name="identify_consultation_type",
                func=identify_consultation_type,
                description="FIRST STEP: Identifies the type of consultation based on the input. Use only when starting or when consultation needs re-identification.",
            ),
            Tool(
                name="identify_lab_tests",
                func=identify_lab_tests,
                description="SECOND STEP: Identifies lab tests using medical knowledge to match against available tests, given doctor's prompt/request. Use only after consultation type confirmation.",
            ),
            Tool(
                name="analyze_patient_history",
                func=analyze_patient_history,
                description="THIRD STEP: Analyzes patient history and suggests additional relevant lab tests. Use only after lab test confirmation and if patient history exists.",
            ),
            Tool(
                name="determine_request_priority",
                func=determine_request_priority,
                description="FOURTH STEP: Determines the priority score for the lab request based on doctor's intent and lab waiting room status. Use after tests are confirmed.",
            ),
            StructuredTool.from_function(
                name="submit_lab_request", 
                func=submit_lab_request,
                description="FINAL STEP: Submits the finalized lab test request to the external API. Use only after all previous steps are confirmed.",
            ),
            StructuredTool.from_function(
                name="escalate_procedure",
                func=escalate_procedure,
                description="Escalates the current procedure to a specialist doctor. Use only when the doctor confirms they want to escalate based on lab results or recommendations."
            )
        ]
        
    def _create_agent(self):
        # Pre-process the physicians JSON to escape curly braces
        physicians_json = json.dumps(self.physicians, indent=2)
        physicians_json = physicians_json.replace("{", "{{").replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 

                f"""
                    You are a friendly and concise medical assistant helping doctors order lab tests. Your goal is to facilitate the lab test ordering process efficiently.
                    This happens in distinct steps. Follow this strict sequence:

                    Workflow Sequence:
                        1. Consultation Type Identification → 2. Lab Test Matching → 3. Patient History Analysis → 4. Submission
                    
                    Escalation Workflow:
                        Physicians list:
                        {physicians_json}

                        Proactive Recommendation:
                            1. Review lab reports in chat history for potential specialist needs
                            2. If certain values or patterns suggest specialist care, proactively recommend escalation
                            3. Suggest appropriate specialists based on the medical findings
                        
                        If the doctor mentions or confirms escalation:
                            1. Verify the doctor's intent to escalate
                            2. If doctor mentions a specialty (e.g., "cardiologist", "endocrinologist"):
                                - Look up physicians with that specialization
                                - Suggest 1-3 appropriate physicians with their SN codes
                            3. If doctor approves a specific physician, use the escalate_procedure tool with their SN
                            4. Provide confirmation when complete

                    Current Step Detection:
                        1. Check chat history to determine which steps are complete
                        2. Proceed to next required step
        
                    Process:
                    Step 1 - Consultation Type:
                        a. Use identify_consultation_type tool to get consultation suggestions.
                        b. If the user's intent at this step is not to make a consultation, politely tell them to first start with choosing the consultation type
                        c. Present suggestion clearly. Make it stand out.
                        d. Present matches with respective ids
                        e. Ask for explicit confirmation
                        f. Repeat until approved

                    Step 2 - Lab Test Identification:
                        a. Use identify_lab_tests tool if consultation type is confirmed
                        b. You should provide the identify_lab_tests tool with the doctor's desire of lab tests to perform.
                        c. Present matches with o_ids
                        d. Ask for confirmation/modifications
                        e. Repeat until approved

                    Step 3 - Patient History Analysis (if available):
                        a. Use analyze_patient_history tool after lab tests are confirmed. The tool receives the patient_history list present in the input data.
                        b. Compare suggested tests with confirmed tests in step 2.
                        c. Present additional suggestions/recommendations with justifications
                        d. Ask which ones to include
                    
                    Step 4 - Priority Determination:
                        a. Use determine_request_priority tool
                        b. Pass the doctor's entire conversation history and lab_waiting_room from input data
                        c. Present the determined priority score and justification
                        d. Ask for confirmation or adjustment
                        e. Explain priority scale (-100 to 100 where 0 is neutral, positive is higher priority, negative is lower)

                    Final Step - Submission (Use this step only when all previous steps are confirmed):
                        a. Extract operation_id and user_id from the input dictionary
                        b. Compile the list of confirmed test_ids (using the "o_id" field)
                        c. Use submit_lab_request tool with format:
                            submit_lab_request(operation_id="actual_id", user_id="user_id", test_ids=["id1", "id2", ...], consultation_type="consultation_id")
                        e. Show success/failure message. If successful, make a report of the whole procedure from step 1 to final step.

                    Escalation Handling:
                        - Proactively review lab reports in the chat history for concerning values or patterns
                        - If you identify issues that require specialist attention, suggest escalation even if doctor hasn't mentioned it
                        - When suggesting escalation, provide:
                            1. Medical reasoning based on specific lab values/findings
                            2. Recommended specialization(s) based on these findings
                            3. List of 1-3 physicians with the relevant specialization (include SN code and specializations)
                        
                        - When the doctor wants to escalate:
                            1. If they mention a specific specialization:
                                - Look up physicians with that specialization from the provided list of physicians
                                - Suggest appropriate specialists with their SN codes
                                - Example: "I found these Gynecologists: Dr. ID: PCD24A03"
                            2. Once they confirm a specific physician:
                                - Use the escalate_procedure tool with the confirmed SN code
                                - Extract the procedure_id from the input dictionary
                                - Use escalate_procedure tool with format:
                                    escalate_procedure(procedure_id, doctor_id)
                                    where procedure_id is the procedure_id from the session, and the doctor_id is the doctor's SN code
                                - Provide confirmation of successful escalation or report any issues
                        
                        - Look for phrases like "escalate to specialist", "refer to [specialty]", "need a [specialty]", etc.
                        - The chat history may contain lab reports with specialist recommendations - be attentive to these

                    State Management:
                        - Maintain natural flow while tracking progress implicitly
                        - Remember previous confirmations in conversation context
                        
                    Important notes:
                        - Never proceed to next step without explicit confirmation
                        - Maintain state through chat history
                        - Always ask for confirmation before submission
                        - Handle one step at a time
                        - Always submit all confirmed lab tests in a single request, and you should submit only their corresponding o_id values
                        - The consultation_type to submit should be the ID value, not the display name
                        - Be concise but thorough in your communications
                        - Priority scores range from -100 to 100, where 0 is neutral
                        - For escalation, the doctor only needs to specify a specialty - you should find matching physicians
                        - Always get final confirmation on a specific physician (with SN code) before escalating
                        - Be proactive in suggesting escalation when lab results indicate specialist care is needed
                    
                    Remember to be friendly and professional while keeping responses brief and to the point.
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
        
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"]
            }
            | prompt
            | self.llm.bind_tools(self.tools)
            | OpenAIToolsAgentOutputParser()
        )

        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        
        return agent_executor
    
    def set_session_context(self, session_data: Dict):
        """Set the global session context for all tools to access"""
        self.session_context = session_data
        
        # Create tools and agent after session context is set
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def process_request(self,
                        chat_type: str, 
                        prompt: str,
                        session_filepath: str) -> Dict:
        """
        Main method to process a doctor's lab test request.

        Args:
            type (str): Type of request
            prompt (str): The user's request text
            session_filepath: The location of this particular session
            
        Returns:
            Dict: Agent's response
        """
        # Load the session data and set the global context
        if session_filepath and os.path.exists(session_filepath):
            session = self.load_session_from_file(session_filepath)
            self.set_session_context(session)
        else:
            return "Error: Session data not found"
        
        user_id = self.session_context.get('user_id')
        procedure_id = self.session_context.get('procedure_id')

        try:
            chat_id = f"{user_id}_{procedure_id}"
            chat_history = self.chat_manager.get_chat_history(chat_id)
        except ValueError:
            # Create new chat session if it doesn't exist
            metadata = {
                "type": chat_type,
            }
            chat_id = self.chat_manager.create_chat_session(
                user_id=user_id,
                procedure_id=procedure_id,
                metadata=metadata
            )
            chat_history = []
        
        procedure_id = self.session_context.get('procedure_id')
        operation_id = self.session_context.get('operation_id')
        input_data = {
            "input": prompt,
            "procedure_id": procedure_id,
            "operation_id": operation_id,
            "user_id": user_id
        }
        
        result = self.agent.invoke({
            "input": input_data,
            "chat_history": chat_history
        })

        # Store the new messages
        self.chat_manager.add_message(
            chat_id=chat_id,
            message=HumanMessage(content=prompt)
        )
        self.chat_manager.add_message(
            chat_id=chat_id,
            message=AIMessage(content=result["output"])
        )
        
        return result["output"]