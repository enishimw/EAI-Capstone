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
            model="gpt-4o-mini"
        )

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.auth_token = auth_token
        
        # Load and process lab tests
        self.consultations = self._load_and_process_consultations(consultations_path)

        # Load and process lab tests
        self.lab_tests = self._load_and_process_tests(lab_tests_path)
        
        self.tools = self._create_tools()
        self.agent = self._create_agent()
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
            Identifies the type of consultation based on the doctor's input.
            Returns the selected consultation type.
            """
            # Get current date info
            current_date = datetime.now()
            is_weekend = current_date.weekday() >= 5  # 5=Saturday, 6=Sunday
            
            prompt = f"""Based on the doctor's request: "{text}"

            Task: Identify the most appropriate consultation type from the available options:
            {json.dumps(self.consultations, indent=2)}

            Current date: {current_date.strftime("%Y-%m-%d")}
            Is weekend/holiday: {"Yes" if is_weekend else "No"}

            Process:
            1. Extract any mention of consultation type (e.g., "new consultation", "follow-up", etc.)
            2. Consider the doctor's specializations and match to relevant consultation types
            3. The consultation type is related uniquely to the doctor's specializations and not on the requested lab test
            3. If today is a weekend/holiday, prefer consultation types containing "weekend, holiday"
            4. Identify the most appropriate consultation type ID
            5. Respond with a short recommendation including the consultation type ID and name

            Example response:
            "I recommend 'New consultation - General practitioner (weekend, holiday)' (ID: 123) based on your request for a new consultation and because today is a weekend."
            
            Remember to be concise but clear.
            """
            response = self.llm.invoke(prompt)
            return response
        
        @tool
        def identify_lab_tests(text: str) -> List[str]:
            """
            Identifies requested lab tests using medical knowledge to match against available tests.
            """
            prompt = f"""Based on the doctor's request: "{text}"

            Task: Identify the specific lab tests needed from our system:
            {json.dumps(self.lab_tests, indent=2)}

            Process:
            1. Extract all mentions of specific lab tests or test categories from the request
            2. Match the doctor's request against available tests considering:
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
            """
            Analyzes patient history and suggests additional relevant lab tests.
            Returns a list of suggested tests with o_ids and reasons.
            """
            try:
                # Try to parse the input if it's a string representation of JSON
                if isinstance(patient_history, str):
                    patient_history = json.loads(patient_history)
            except:
                return "Error: Could not parse patient history data"

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
            2. [...]"""

            response = self.llm.invoke(prompt).content
            return response

        def submit_lab_request(operation_id: str, user_id:str, test_ids: List[str], consultation_type: str) -> Dict:
            """
            Submits the finalized lab test request to the external API.
            """
            api_endpoint = "https://x.clinicplus.pro/api/llm/consultations/lab_request"
            
            # Prepare the payload
            payload = {
                "operation": operation_id,
                "doctor": user_id,
                "consultation": consultation_type,
                "exams": {"exams": test_ids},
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

            print("submit_lab_request_response: ", response.text)

            return json.loads(response.text)
            
        return [
            Tool(
                name="identify_consultation_type",
                func=identify_consultation_type,
                description="Identifies the type of consultation based on the input.",
            ),
            Tool(
                name="identify_lab_tests",
                func=identify_lab_tests,
                description="Identifies requested lab tests using medical knowledge to match against available tests.",
            ),
            Tool(
                name="analyze_patient_history",
                func=analyze_patient_history,
                description="Analyzes patient history and suggests additional relevant lab tests.",
            ),
            StructuredTool.from_function(
                name="submit_lab_request", 
                func=submit_lab_request,
                description="Submits the finalized lab test request to the external API.",
            )
        ]
        
    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                """
                    You are a friendly and concise medical assistant helping doctors order lab tests. Your goal is to facilitate the lab test ordering process efficiently.
        
                    Process:
                    1. For each new request:
                    - Check if the doctor has mentioned both consultation type AND lab tests together
                    - If only lab tests are mentioned, use identify_consultation_type tool to determine the appropriate consultation type
                    - If only consultation type is mentioned, ask for which lab tests are needed
                    - If both are mentioned, use both tools to confirm the correct interpretation
                    - Analyse the patient history and provide suggestion for additional tests, if applicable
                    - Present all suggestions clearly and ask for confirmation

                    2. For consultation type identification:
                    - Use identify_consultation_type tool 
                    - Present the suggested consultation type to the doctor clearly
                    - Confirm before proceeding
                    
                    3. For lab test identification:
                    - Use identify_lab_tests tool
                    - Present the matches clearly and ask for confirmation
                    - Make any adjustments requested by the doctor

                    4. For patient history analysis:
                    - Use analyze_patient_history tool when patient history is available.
                    - The tool receives the patient_history list present in the input data.
                    - Compare suggested tests with doctor's requested tests
                    - Present additional recommendations with justifications
                    - Ask if doctor wants to add any suggested tests
                        
                    5. Once all details are confirmed:
                    - Extract operation_id and user_id from the input dictionary
                    - Compile the list of confirmed test_ids (using the "o_id" field)
                    - Submit the request using submit_lab_request with format:
                        submit_lab_request(operation_id="actual_id", user_id="user_id", test_ids=["id1", "id2"], consultation_type="consultation_id")
                    - Confirm successful submission to the doctor

                    6. When finalizing:
                    - Include both confirmed and approved suggested tests
                    - Ensure all test IDs are valid o_ids
                        
                    Important notes:
                    - Always handle both consultation type and lab tests even when provided in a single request
                    - Process both pieces of information in parallel when given together
                    - Always submit all lab tests in a single request, and you should submit only their corresponding ID value
                    - The consultation_type to submit should be the ID value, not the display name
                    - Be concise but thorough in your communications
                    - If it's a weekend or holiday, recommend the appropriate consultation variant
                    
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
        
    def process_request(self,
                        type: str, 
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

        try:
            chat_id = f"{user_id}_{procedure_id}"
            chat_history = self.chat_manager.get_chat_history(chat_id)
        except ValueError:
            # Create new chat session if it doesn't exist
            metadata = {
                "type": type,
            }
            chat_id = self.chat_manager.create_chat_session(
                user_id=user_id,
                procedure_id=procedure_id,
                metadata=metadata
            )
            chat_history = []
        
        
        operation_id = session.get('operation_id')
        doctor_specializations = session.get('doctor_specializations')
        patient_history = session.get('patient_history')

        input_data = {
            "input": prompt,
            "operation_id": operation_id,
            "user_id": user_id,
            "doctor_specializations": doctor_specializations,
            "patient_history": patient_history
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