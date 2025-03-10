import os
import json
from typing import List, Dict, Any, Union, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import tool, Tool

from ChatHistoryManager import ChatHistoryManager

class LabResultAnalysisAgent:
    def __init__(self, api_key: str, lab_tests_path: str, chat_history_path: str = "chat_histories/"):
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4o"
        )
        
        # Load the lab tests with normal ranges
        self.lab_tests = self._load_lab_tests(lab_tests_path)
        
        # Initialize session context to store patient results
        self.session_context = None
        
        # Initialize chat history manager
        self.chat_manager = ChatHistoryManager(storage_path=chat_history_path)
        
        # Initialize tools and agent
        self.tools = None
        self.agent = None
    
    def _load_lab_tests(self, lab_tests_path: str) -> Dict:
        """
        Load lab tests from JSON file and prepare a lookup dictionary
        """
        with open(lab_tests_path, 'r') as f:
            lab_tests = json.load(f)
        
        # Create a lookup dictionary by test ID
        test_lookup = {}
        for test in lab_tests:
            test_lookup[test['o_id']] = test
        
        return test_lookup
    
    def load_session_from_file(self, filepath: str) -> Dict:
        """
        Load session data from a JSON file.
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_session_to_file(self, filepath: str, session_data: Dict) -> None:
        """
        Save session data to a JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def set_session_context(self, session_data: Dict):
        """Set the global session context for all tools to access"""
        self.session_context = session_data
        
        # Create tools and agent after session context is set
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        @tool
        def analyze_result(test_id: str, result_value: Any) -> str:
            """
            Analyzes a single lab test result for abnormalities.
            
            Args:
                test_id: The ID of the lab test
                result_value: The result value entered by the lab technician
                
            Returns:
                Analysis of the result including whether it's normal, critical, or abnormal
            """
            # Get the test information
            test = self.lab_tests.get(int(test_id))
            if not test:
                return f"Error: Test ID {test_id} not found in the database."
            
            # Get the current session results to provide context
            current_results = self.session_context.get('results', {})
            patient_info = self.session_context.get('patient_info', {})
            
            # Update the session with this new result
            if 'results' not in self.session_context:
                self.session_context['results'] = {}
            
            self.session_context['results'][test_id] = result_value
            
            # Prepare prompt for analysis
            prompt = f"""
            Analyze this lab test result:
            
            Test: {test.get('locale_name', 'Unknown test')} (ID: {test_id})
            Result value: {result_value}
            
            Patient information:
            {json.dumps(patient_info, indent=2)}
            
            Other recent results:
            {json.dumps(current_results, indent=2)}
            
            Please provide:
            1. Whether this result is normal, abnormal, or critical
            2. Clinical significance of this result
            3. Possible actions for the lab technician to take (if any)
            
            Respond in a professional tone suitable for a lab technician.
            """
            
            # Get analysis from LLM
            response = self.llm.invoke(prompt)
            return response.content
        
        @tool
        def get_comprehensive_analysis() -> str:
            """
            Provides a comprehensive analysis of all test results for the current session.
            
            Returns:
                Comprehensive analysis of all test results, highlighting patterns, critical values, and recommendations
            """
            results = self.session_context.get('results', {})
            patient_info = self.session_context.get('patient_info', {})
            
            prompt = f"""
            Provide a comprehensive analysis of the following lab test results:
            
            Patient information:
            {json.dumps(patient_info, indent=2)}
            
            Test results:
            {json.dumps(results, indent=2)}

            Lab tests for normal ranges reference:
            {json.dumps(self.lab_tests, indent=2)}
            
            Please include:
            1. Overall assessment of the patient's test results
            2. Highlight any critical or abnormal values
            3. Identify patterns or relationships between test results
            4. Suggest possible clinical implications based on the constellation of results
            6. Note any quality control issues or possible interferences that may affect result interpretation
            
            Format your analysis in a clear, professional manner for a lab technician or clinician.
            """
            
            # Get comprehensive analysis from LLM
            response = self.llm.invoke(prompt)
            return response.content
        
        @tool
        def identify_critical_values() -> str:
            """
            Identifies only the critical values that require immediate attention.
            
            Returns:
                List of critical values and recommended immediate actions
            """
            results = self.session_context.get('results', {})
            
            # Prepare the test results with test names for better context
            named_results = {}
            for test_id, value in results.items():
                test_id_int = int(test_id) if test_id.isdigit() else test_id
                test = self.lab_tests.get(test_id_int, {})
                test_name = test.get('locale_name', f"Unknown test ({test_id})")
                named_results[test_name] = value
            
            prompt = f"""
            Review these lab test results and identify ONLY critical values that require immediate clinical attention:
            
            Test results:
            {json.dumps(named_results, indent=2)}
            
            For each critical value:
            1. Specify the test name and value
            2. Indicate the typical critical threshold
            3. Recommend immediate actions for the lab technician
            4. Note if any immediate notifications to clinical staff are required
            
            If there are no critical values, clearly state that no critical values were identified.
            
            Provide your response in a concise, prioritized format.
            """
            
            # Get critical values analysis from LLM
            response = self.llm.invoke(prompt)
            return response.content
        
        return [
            Tool(
                name="analyze_result",
                func=analyze_result,
                description="Analyzes a single lab test result for abnormalities. Use this when a new result is entered."
            ),
            Tool(
                name="get_comprehensive_analysis",
                func=get_comprehensive_analysis,
                description="Provides a comprehensive analysis of all test results for the current session. Use this to get an overview of the patient's results."
            ),
            Tool(
                name="identify_critical_values",
                func=identify_critical_values,
                description="Identifies only the critical values that require immediate attention. Use this to quickly spot urgent issues."
            ),
        ]
    
    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                """
                You are a knowledgeable laboratory assistant helping lab technicians analyze and interpret lab test results.
                Your goal is to identify abnormal values, critical results, and patterns that require attention.
                
                When a lab technician enters a new result:
                1. Use the analyze_result tool to evaluate the individual result
                2. Highlight if the result is normal, abnormal, or critical
                3. Provide context and clinical significance
                
                When reviewing multiple results:
                1. Use the get_comprehensive_analysis tool to evaluate all results together
                2. Identify patterns and relationships between different test results
                3. Highlight critical values using the identify_critical_values tool
                
                Important guidelines:
                - Be concise and focused in your communications
                - Provide context for abnormal values (typical ranges, clinical significance)
                - When uncertain, acknowledge limitations
                - Prioritize critical values that require immediate attention
                - Use appropriate medical terminology
                - Consider how different test results might interrelate
                - Remember that you're assisting a lab technician, not a physician
                
                Results can be of various types:
                - Numeric values with units (e.g., 5.2 mg/dL)
                - Qualitative results (e.g., Positive/Negative)
                - Descriptive findings (e.g., "Moderate presence of...")
                
                Adapt your analysis to the type of result provided.
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
    
    def process_result(self, 
                      test_id: str, 
                      result_value: Any, 
                      session_filepath: str) -> Dict:
        """
        Process a lab test result and provide analysis.
        
        Args:
            test_id: The ID of the lab test
            result_value: The result value entered by the lab technician
            session_filepath: Path to the session file
            
        Returns:
            Dict: Agent's response
        """
        # Load or initialize the session data
        if os.path.exists(session_filepath):
            session = self.load_session_from_file(session_filepath)
        else:
            session = {
                'results': {},
                'patient_info': {}
            }
        
        # Update session with the new result
        if 'results' not in session:
            session['results'] = {}
        
        session['results'][test_id] = result_value
        self.set_session_context(session)
        
        # Save the updated session
        self.save_session_to_file(session_filepath, session)
        
        # Extract user_id and procedure_id from session filepath for chat history
        filename = os.path.basename(session_filepath)
        file_parts = os.path.splitext(filename)[0].split('_')
        
        if len(file_parts) >= 2:
            user_id = file_parts[0]
            procedure_id = file_parts[1]
            
            try:
                chat_id = f"{user_id}_{procedure_id}"
                chat_history = self.chat_manager.get_chat_history(chat_id)
            except ValueError:
                # Create new chat session if it doesn't exist
                metadata = {
                    "type": "lab_result_analysis",
                }
                chat_id = self.chat_manager.create_chat_session(
                    user_id=user_id,
                    procedure_id=procedure_id,
                    metadata=metadata
                )
                chat_history = []
        else:
            chat_history = []
        
        # Prepare the input for the agent
        input_data = {
            "test_id": test_id,
            "result_value": result_value
        }
        
        # Get the result from the agent
        result = self.agent.invoke({
            "input": input_data,
            "chat_history": chat_history
        })
        
        # Store the new messages if chat history is available
        if 'chat_id' in locals():
            self.chat_manager.add_message(
                chat_id=chat_id,
                message=HumanMessage(content=f"New result: Test ID {test_id}, Value: {result_value}")
            )
            self.chat_manager.add_message(
                chat_id=chat_id,
                message=AIMessage(content=result["output"])
            )
        
        return result["output"]
    
    def process_request(self,
                        chat_type: str,
                        prompt: str,
                        session_filepath: str) -> Dict:
        """
        Process a general request from the lab technician.
        
        Args:
            chat_type: Type of chat
            prompt: The user's prompt or request
            session_filepath: Path to the session file
            
        Returns:
            Dict: Agent's response
        """
        # Load the session data
        if os.path.exists(session_filepath):
            session = self.load_session_from_file(session_filepath)
            self.set_session_context(session)
        else:
            return "Error: Session data not found"
        
        # Extract user_id and procedure_id for chat history
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
        
        input_data = {
            "input": prompt,
            "results": self.session_context.get('results'),
        }
        
        # Get result from the agent
        result = self.agent.invoke({
            "input": input_data,
            "chat_history": chat_history
        })
        
        # Store the new messages if chat history is available
        if 'chat_id' in locals():
            self.chat_manager.add_message(
                chat_id=chat_id,
                message=HumanMessage(content=prompt)
            )
            self.chat_manager.add_message(
                chat_id=chat_id,
                message=AIMessage(content=result["output"])
            )
        
        return result["output"]