import os
import ast
import json
import requests
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
    def __init__(self, api_key: str, lab_tests_path: str, auth_token: str = None, chat_history_path: str = "chat_histories/"):
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4o"
        )

        # Store auth token for API requests
        self.auth_token = auth_token
        
        # Load the lab tests with normal ranges
        self.lab_tests = self._load_lab_tests(lab_tests_path)
        
        # Initialize session context to store patient results
        self.session_context = None
        
        # Initialize chat history manager
        self.chat_manager = ChatHistoryManager(storage_path=chat_history_path)
        self.physicians_path = "physicians.json"
        
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
        def analyze_result(lab_test_result: Any) -> str:
            """
            Analyzes a single lab test result for abnormalities.
            
            Args:
                lab_test_result: The lab test and the corresponding result entered by the lab technician
                
            Returns:
                Analysis of the result including a numerical representation of abnormality and brief insight
            """
            
            # Get the current session results to provide context
            patient_info = self.session_context.get('patient_info', {})

            # Convert the string to a dictionary
            lab_test_result = json.loads(lab_test_result.replace("'", '"'))

            # Extract test information
            test_id = None
            test_value = None
            test_name = None
            test_units = None

            try:
                # Extract the relevant test information based on the structure
                if isinstance(lab_test_result, dict):
                    # If it's a direct test result
                    if 'o_id' in lab_test_result:
                        test_id = lab_test_result.get('o_id')
                        test_name = lab_test_result.get('locale_name', 'Unknown Test')
                        # Extract value from the result field if it exists
                        result = lab_test_result.get('result', {})
                        if isinstance(result, dict):
                            test_value = result.get('value')
                            # For qualitative results like positive/negative
                            if test_value is None and 'option' in result:
                                test_value = result.get('option')
                    # Another possible structure
                    elif 'test_id' in lab_test_result:
                        test_id = lab_test_result.get('test_id')
                        test_name = lab_test_result.get('test_name', 'Unknown Test')
                        test_value = lab_test_result.get('value')
            except Exception as e:
                print(f"Error extracting test information: {str(e)}")
            
            # Find the specific test in lab_tests reference data
            test_reference = None
            test_normal_ranges = []
            
            if test_id:
                # Convert to int if it's a string of digits
                test_id_int = int(test_id) if isinstance(test_id, str) and test_id.isdigit() else test_id
                
                # Look up the test in the reference data
                if test_id_int in self.lab_tests:
                    test_reference = self.lab_tests[test_id_int]
                    test_name = test_reference.get('locale_name', test_name)
                    test_units = test_reference.get('unit_of_measure')
                    test_normal_ranges = test_reference.get('normal_values', [])
            
            # Prepare a simplified reference object for the LLM
            simplified_reference = {
                'test_id': test_id,
                'test_name': test_name,
                'unit_of_measure': test_units,
                'normal_ranges': test_normal_ranges
            }
            
            # Prepare prompt for analysis
            prompt = f"""
            Analyze this lab test result:
            
            Test Information:
                - Name: {simplified_reference['test_name']}
                - ID: {simplified_reference['test_id']}
                - Value: {test_value}
                - Units: {simplified_reference['unit_of_measure'] if simplified_reference['unit_of_measure'] else 'Not specified'}
            
            Reference Normal Ranges:
                {json.dumps(simplified_reference['normal_ranges'], indent=2)}
            
            Patient information:
                {json.dumps(patient_info, indent=2)}
            
            Please analyze this lab result and provide a proper JSON object (not a string) with exactly these three fields (always use the lab tests for normal ranges reference):
                {{
                    "abnormality_score": <number>,
                    "insight": <string>,
                    "analysis": <string>
                }}
            
            Where:
                - abnormality_score is a number indicating how far the result is from the normal range:
                    * 1: dangerously above normal range
                    * 0.5: acceptably above normal range
                    * 0: within normal range or not applicable
                    * -0.5: acceptably below normal range
                    * -1: dangerously below normal range
                - insight is a concise one-phrase description/insight about the result
                - analysis is a more detailed explanation including clinical significance and possible actions

            Important:
                - The abnormality_score should be based directly on the reference normal ranges provided
                - If no normal ranges are provided or the test is qualitative (like positive/negative), use your medical knowledge to assess abnormality
                - Consider patient information if normal ranges vary by age, gender, etc.
                - Be precise in your assessment
                - Don't guess normal ranges if they aren't provided
            
            Your response must be valid JSON with these exact fields and nothing else. Do not include markdown formatting, code blocks, or any explanatory text. JUST RETURN THE RAW JSON OBJECT AND NOTHING ELSE.
            """
            
            # Get analysis from LLM
            response = self.llm.invoke(prompt)
            # Parse the JSON from the response
            try:
                # Try to parse the content as JSON directly
                result = json.loads(response.content)
                # Ensure the required fields are present
                required_fields = ["abnormality_score", "insight", "analysis"]
                for field in required_fields:
                    if field not in result:
                        result[field] = f"Missing {field}"
                
                # Return the parsed JSON as a string that can be parsed again later
                return json.dumps(result)
            except json.JSONDecodeError:
                # If parsing fails, return a formatted error message as JSON
                error_response = {
                    "abnormality_score": 0, 
                    "insight": "Error processing result",
                    "analysis": "The analysis tool encountered an error parsing the response. Please review the lab result manually."
                }
                return json.dumps(error_response)
        
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

            # Save the analysis to the session context for later use
            self.session_context['result_analysis'] = response.content
            
            # Also save back to file
            user_id = self.session_context.get('user_id')
            procedure_id = self.session_context.get('procedure_id')
            session_filepath = f"sessions/{user_id}_{procedure_id}_lab_session.json"
            if os.path.exists(session_filepath):
                with open(session_filepath, 'r') as f:
                    session_data = json.load(f)
                
                session_data['result_analysis'] = response.content
                
                with open(session_filepath, 'w') as f:
                    json.dump(session_data, f, indent=2)
            
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

        @tool
        def escalate_procedure(self, procedure_id: str, doctor_id: str) -> str:
            """
            Escalates a procedure to a specialist doctor by making a POST request to the API.
            
            Args:
                procedure_id: ID of the procedure to escalate
                doctor_id: ID of the specialist doctor to escalate the procedure to
                
            Returns:
                String response indicating success or failure of the escalation request
            """
            try:
                api_endpoint = "https://x.clinicplus.pro/api/llm/consultations/escalate"
                
                # Prepare the payload
                payload = {
                    "procedure_id": procedure_id,
                    "doctor_id": doctor_id
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
                
                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success", False):
                        return f"Successfully escalated procedure {procedure_id} to doctor {doctor_id}."
                    else:
                        error_message = data.get("message", "Unknown error")
                        return f"API returned error: {error_message}"
                else:
                    return f"API request failed with status code {response.status_code}: {response.text}"
            
            except Exception as e:
                return f"An error occurred while escalating the procedure: {str(e)}"

        return [
            Tool(
                name="analyze_result",
                func=analyze_result,
                description="Analyzes a single lab test result for abnormalities."
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
                - When using the analyze_result tool, return its output directly without additional formatting. Do not wrap JSON responses in code blocks or add explanatory text around them
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
    
    def generate_final_report(self, lab_tech_comment: str) -> str:
        """
        Generates a final report combining the saved comprehensive analysis with lab tech comments.
        
        Args:
            lab_tech_comment: Comments from the lab technician
            
        Returns:
            Formatted report for the doctor
        """
        # Get the saved comprehensive analysis
        comprehensive_analysis = self.session_context.get('result_analysis', None)
        
        # If no saved analysis is found, indicate this in the report
        if not comprehensive_analysis:
            comprehensive_analysis = "Note: No comprehensive analysis was previously generated during the lab session."
        
        # Get the patient info for context
        patient_info = self.session_context.get('patient_info', {})

        physicians = []
        if os.path.exists(self.physicians_path):
            with open(self.physicians_path, 'r') as f:
                physicians = json.load(f)
        
        prompt = f"""
        Create a formal lab results report combining comprehensive analysis with lab technician comments.
        
        Patient information:
        {json.dumps(patient_info, indent=2)}
        
        Lab technician's comments:
        {lab_tech_comment}
        
        Comprehensive analysis:
        {comprehensive_analysis}

        Physicians and their specializations:
        {json.dumps(physicians, indent=2)}
        
        Please format this into a professional report with:
        1. A concise summary at the top highlighting key findings and critical values
        2. The lab tech's professional observations and comments
        3. Detailed and well organized analysis of test results
        4. Any patterns or correlations observed across multiple tests
        5. Follow-up recommendations if appropriate
        6. Physician escalation recommendations if the results indicate a need for specialist consultation
            - Review the test results and determine if a specialist consultation is needed
            - If specialist consultation is needed, recommend specific physicians based on their specializations
            - Match the relevant medical specialization to the abnormal test findings
            - Include the physician's ID (sn) and specialization in the recommendation
        
        The report should be clear, professional, and focus on the most clinically relevant information.
        If no specialist consultation is needed based on the results, simply omit that section.
        """
        
        # Get the final report from LLM
        response = self.llm.invoke(prompt)
        return response.content

    def process_result(self, 
                        chat_type: str,
                        prompt: str,
                        lab_test_result: Dict,
                        session_filepath: str) -> Dict:
        """
        Process a lab test result and provide analysis.
        
        Args:
            lab_test_result: The lab test and the corresponding result entered by the lab technician
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
        
        # Prepare the input for the agent
        input_data = {
            "input": prompt,
            "result": lab_test_result
        }
        
        # Get the result from the agent
        result = self.agent.invoke({
            "input": input_data,
            "chat_history": []
        })
        
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
        
        if chat_type != "lab_result_analysis":
            self.chat_manager.add_message(
                chat_id=chat_id,
                message=HumanMessage(content=prompt)
            )
        
        self.chat_manager.add_message(
            chat_id=chat_id,
            message=AIMessage(content=result["output"])
        )
        
        return result["output"]