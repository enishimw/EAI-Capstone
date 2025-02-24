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
            prompt = f"""I am a medical assistant helping you with lab test orders. Based on your request: "{text}"

            I'll analyze our available consultation types and suggest matches from our system:
            {json.dumps(self.consultations, indent=2)}
            -------
            Let me help identify the specific consultation type you need:
            1. I'll match your request against our available consultation types considering:
            - Standard medical terminology and common abbreviations
            - The current date {datetime.now().strftime("%Y-%m-%d")}, to choose the right variant, if it's in the weekend then the "weekend/holiday" variant is more appropriate
            - the doctor's specializations
            - Different naming conventions for the same consultation type
            2. I'll list the matching consultation types I've found
            3. Please confirm if this is the consultation type you want to proceed with
            4. If you need any clarification or have additional consultation types in mind, please let me know

            Which of these consultation types would you like to proceed with?
            """
            response = self.llm.invoke(prompt)
            return response
        
        @tool
        def identify_lab_tests(text: str) -> List[str]:
            """
            Identifies requested lab tests using medical knowledge to match against available tests.
            """
            prompt = f"""I am a medical assistant helping you with lab test orders. Based on your request: "{text}"

            I'll analyze our available tests and suggest matches from our system:
            {json.dumps(self.lab_tests, indent=2)}
            -------
            Note that some tests are children of others.

            Let me help identify the specific tests you need:
            1. I'll match your request against our available tests considering:
               - Standard medical terminology and common abbreviations
               - Different naming conventions for the same test
               - Related or complementary tests
            2. I'll list the matching tests I've found
            3. Please confirm if these are the tests you want to order
            4. If you need any clarification or have additional tests in mind, please let me know

            Which of these tests would you like me to order for your patient?
            """
            response = self.llm.invoke(prompt)

            return response
            
        @tool
        def analyze_patient_history(history: Dict) -> List[str]:
            """
            Analyzes patient history and suggests additional relevant lab tests.
            Returns a list of suggested test IDs.
            """
            # Implement logic to analyze patient history and suggest tests
            suggested_tests = []
            return suggested_tests

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
            # Tool(
            #     name="analyze_patient_history",
            #     func=analyze_patient_history,
            #     description="Analyzes patient history and suggests additional relevant lab tests.",
            # ),
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
                    You are a medical assistant helping doctors order lab tests.
                
                    Process:
                    1. When you receive a lab test request:
                    - First, ask the doctor to specify the consultation type using identify_consultation_type tool
                    - Extract the consultation_id from the input dictionary to be used later in the submit_lab_request tool
                    - Present the matches to the doctor for confirmation
                    - Ask for clarification if needed

                    2. After that:
                    - Use identify_lab_tests tool to analyze and match the test request
                    - Present the matches to the doctor for confirmation
                    - Ask for clarification if needed
                    
                    2. After receiving confirmation:
                    - Extract operation_id from the input dictionary
                    - Create a list of confirmed test_ids. These should be the "o_id" field on the actual test.
                    - Call submit_lab_request with EXACTLY this format:
                        submit_lab_request(operation_id="actual_id", user_id="user_id" test_ids=["id1", "id2"], consultation_type="consultation_id")
                    - Provide confirmation of successful submission
                    
                    Important: The submit_lab_request tool requires:
                    - operation_id (string): Available in the input dictionary
                    - user_id (string): Available in the input dictionary
                    - test_ids (list of strings): List of confirmed test IDs
                    - consultation_type (string): The confirmed consultation type

                    IMPORTANT: Always submit the lab tests in one go as a list, not as separate requests.
                    
                    Always:
                    - Be clear about which tests you've identified
                    - Ask for explicit confirmation before submitting
                    - Handle any clarifications or modifications requested
                    - Confirm successful submission with the doctor
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