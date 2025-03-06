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
    def __init__(self, api_key: str, auth_token: str, consultations_path: str, lab_tests_path: str, chat_history_path: str = "chat_histories/"):
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4o"
        )

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.auth_token = auth_token
        
        # Load and process consultations
        self.consultations = self._load_and_process_consultations(consultations_path

        # Load and process lab tests
        self.lab_tests = self._load_and_process_tests(lab_tests_path)
        
        self.tools = self._create_tools()
        self.agent = self._create_agent()
=======
        # Load and process lab tests
        self.lab_tests = self._load_and_process_tests(lab_tests_path)

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
            
            prompt = f"""Based on the doctor's specializations: "{json.dumps(self.doctor_specializations, indent=2)}"
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

        def analyze_patient_history(patient_history: str) -> List[str]:
=======
        def analyze_patient_history() -> List[str]:
>>>>>>> 524e92b85689dd94833b94f0e40d44ff78735b30
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


        def determine_request_priority(doctor_intent: str, lab_waiting_room: list) -> Dict
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
            
            # Check how busy the lab is
            lab_load = len(lab_waiting_room) if lab_waiting_room else 0
            
            prompt = f"""
            Analyze the doctor's message and lab waiting room status to determine a priority score.
            
            Doctor's message: "{doctor_intent}"

            Current lab load: {lab_load} pending test(s)

            Current lab load: "{json.dumps(self.session_context.get('lab_waiting_room', []), indent=2)}"

            
            Task: Calculate a priority score from -100 to 100 where:
            - 0 is the default neutral priority
            - 1 to 100 indicates increasing urgency (100 being highest emergency priority)
            - -1 to -100 indicates decreasing urgency (-100 being lowest priority/can wait)
            
            Consider these factors:
            1. Explicit urgency keywords (STAT, urgent, emergency, ASAP, critical, etc.)
            2. Medical conditions suggesting urgency (cardiac symptoms, severe pain, etc.)

            3. Current lab load ({lab_load} pending tests)
=======
            3. Current lab load
>>>>>>> 524e92b85689dd94833b94f0e40d44ff78735b30
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

        # Now update the submit_lab_request tool to include the priority score
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

            Structure
            Tool(

                name="determine_request_priority",
                func=determine_request_priority,
                description="FOURTH STEP: Determines the priority score for the lab request based on doctor's intent and lab waiting room status. Use after tests are confirmed.",
            ),
            StructuredTool.from_function(
                name="submit_lab_request", 
                func=submit_lab_request,
                description="FINAL STEP: Submits the finalized lab test request to the external API. Use only after all previous steps are confirmed.",
            )
        ]
        
    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                """
                    You are a friendly and concise medical assistant helping doctors order lab tests. Your goal is to facilitate the lab test ordering process efficiently.
                    This happens in distinct steps. Follow this strict sequence:

                    Workflow Sequence:
                    1. Consultation Type Identification → 2. Lab Test Matching → 3. Patient History Analysis → 4. Submission

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
<<<<<<< HEAD
        
    def process_request(self,
                        type: str, 
=======
    
    def set_session_context(self, session_data: Dict):
        """Set the global session context for all tools to access"""
        self.session_context = session_data
        
        # Create tools and agent after session context is set
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def process_request(self,
                        chat_type: str
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

        if session_filepath and os.path.exists(session_filepath):
            session = self.load_session_from_file(session_filepath)
        
        user_id = session.get('user_id')
        procedure_id = session.get('procedure_id')

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
                      
                "type": type,
                "type": chat_type,
            }
            chat_id = self.chat_manager.create_chat_session(
                user_id=user_id,
                procedure_id=procedure_id,
                metadata=metadata
            )
            chat_history = []
        

        
        operation_id = session.get('operation_id')
        self.doctor_specializations = session.get('doctor_specializations')
        patient_history = session.get('patient_history')
        lab_waiting_room = session.get('lab_waiting_room', [])

        input_data = {
            "input": prompt,
            "operation_id": operation_id,
            "user_id": user_id,
            "doctor_specializations": self.doctor_specializations,
            "patient_history": patient_history,
            "lab_waiting_room": lab_waiting_room

        operation_id = self.session_context.get('operation_id')
        input_data = {
            "input": prompt,
            "operation_id": operation_id,
            "user_id": user_id

        }
        
        result = self.agent.invoke({
            "input":input_data,
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