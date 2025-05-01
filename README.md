# Intelligent Laboratory Management System

## Overview
The Intelligent Laboratory Management System is a comprehensive platform that leverages Large Language Models (LLMs) to streamline laboratory workflows in medical settings. It bridges the communication gap between doctors and lab technicians through automated lab test ordering, result analysis, and clinical recommendation generation.

By integrating OpenAI's GPT-4o with a custom-built framework, the system demonstrates remarkable capabilities in result interpretation, abnormality detection, and specialist recommendation, achieving high accuracy in clinical analysis.

## Key Features

### For Doctors
- **Natural Language Test Ordering**: Request lab tests using natural language
- **Test Recommendation**: Get suggestions based on patient symptoms and history
- **Patient History Integration**: Receive additional test recommendations based on medical history
- **Prioritization System**: Automatic determination of test urgency
- **Comprehensive Result Analysis**: Receive detailed analysis of lab results with clinical context
- **Specialist Referral**: Get recommendations for specialist consultations based on test results

### For Lab Technicians
- **Test Result Entry**: Input test results with real-time analysis
- **Abnormality Detection**: Automatic identification of values outside normal ranges
- **Critical Value Alerts**: Immediate highlighting of values requiring urgent attention
- **Comprehensive Analysis**: Generation of holistic analysis across multiple tests
- **Report Generation**: Creation of detailed reports for doctors with clinical implications

## System Architecture
The system consists of two primary components:
1. **Lab Test Agent**: Assists doctors in ordering appropriate lab tests
2. **Lab Result Analysis Agent**: Helps lab technicians interpret results and generate reports

## Technical Requirements

### Software Requirements
- Python 3.8 or higher
- Flask web framework
- OpenAI API key
- LangChain framework
- Dependencies listed in `requirements.txt`

### Hardware Requirements
- Minimum 4GB RAM
- 2 CPU cores
- 20GB storage
- Stable internet connection

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/intelligent-lab-management
cd intelligent-lab-management
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
AUTH_TOKEN=your_auth_token
```

### 5. Run the Application
```bash
flask run
```

## API Endpoints

The system provides the following RESTful API endpoints:

- `/initialize-session`: Start a new doctor or lab technician session
- `/initialize-lab-session`: Initialize a lab technician session
- `/process-prompt`: Process doctor's lab test requests
- `/process-lab-prompt`: Handle lab technician queries
- `/process-lab-result`: Process individual lab test results
- `/get-comprehensive-analysis`: Generate analysis of all test results
- `/submit-final-report`: Generate and store the final lab report

## Data Structure

The system relies on several JSON data files:
- `lab_tests.json`: Database of tests with normal ranges
- `physicians.json`: Specialist information for referrals
- `consultation_types.json`: Available consultation formats

## Usage Guide

### Doctor Workflow
1. Start a conversation with the system
2. Select a consultation type
3. Specify required tests or describe symptoms
4. Review additional test suggestions
5. Confirm test selection
6. Receive and analyze lab reports when ready
7. Review specialist recommendations if applicable

### Lab Technician Workflow
1. View incoming test requests
2. Enter test results
3. Review real-time analysis of abnormal values
4. Generate comprehensive analysis
5. Add professional comments
6. Submit final report to the requesting doctor

## Evaluation
The system has been evaluated by medical professionals with positive feedback on test recognition, result analysis, and abnormality detection. Areas for improvement include workflow efficiency and patient history integration.

## Contributors
- [Your Name]
- [Team Member 1]
- [Team Member 2]

## License
[Specify your license here]

## Acknowledgments
- OpenAI for GPT-4o
- LangChain for the framework
- Medical professionals who evaluated the system
