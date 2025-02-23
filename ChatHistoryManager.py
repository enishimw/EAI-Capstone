from typing import List, Dict, Optional
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from datetime import datetime
import json
import os

class ChatHistoryManager:
    def __init__(self, storage_path: str = "chat_histories/"):
        """
        Initialize the ChatHistoryManager with a storage path.
        
        Args:
            storage_path (str): Directory path where chat histories will be stored
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
    def _generate_chat_id(self, user_id: str, procedure_id: str) -> str:
        """
        Generate a unique chat ID combining user and procedure IDs.
        """
        return f"{user_id}_{procedure_id}"
        
    def _get_chat_filepath(self, chat_id: str) -> str:
        """
        Get the full filepath for a chat history file.
        """
        return os.path.join(self.storage_path, f"{chat_id}.json")
        
    def create_chat_session(self, user_id: str, procedure_id: str, metadata: Optional[Dict] = None) -> str:
        """
        Create a new chat session for a user and procedure.
        
        Args:
            user_id (str): Unique identifier for the user
            procedure_id (str): Unique identifier for the medical procedure
            metadata (Dict, optional): Additional metadata about the chat session
            
        Returns:
            str: The generated chat_id
        """
        chat_id = self._generate_chat_id(user_id, procedure_id)
        
        chat_data = {
            "chat_id": chat_id,
            "user_id": user_id,
            "procedure_id": procedure_id,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "messages": []
        }
        
        with open(self._get_chat_filepath(chat_id), 'w') as f:
            json.dump(chat_data, f, indent=2)
            
        return chat_id
        
    def add_message(self, chat_id: str, message: BaseMessage) -> None:
        """
        Add a new message to the chat history.
        
        Args:
            chat_id (str): The chat session identifier
            message (BaseMessage): The message to add (HumanMessage or AIMessage)
        """
        filepath = self._get_chat_filepath(chat_id)
        
        if not os.path.exists(filepath):
            raise ValueError(f"Chat session {chat_id} does not exist")
            
        with open(filepath, 'r') as f:
            chat_data = json.load(f)
        
        # Convert message to serializable format
        message_data = {
            "type": message.__class__.__name__,
            "content": message.content,
            "timestamp": datetime.now().isoformat()
        }
        
        chat_data["messages"].append(message_data)
        
        with open(filepath, 'w') as f:
            json.dump(chat_data, f, indent=2)
            
    def get_chat_history(self, chat_id: str) -> List[BaseMessage]:
        """
        Retrieve the chat history for a given chat session.
        
        Args:
            chat_id (str): The chat session identifier
            
        Returns:
            List[BaseMessage]: List of messages in the chat history
        """
        filepath = self._get_chat_filepath(chat_id)
        
        if not os.path.exists(filepath):
            raise ValueError(f"Chat session {chat_id} does not exist")
            
        with open(filepath, 'r') as f:
            chat_data = json.load(f)
            
        # Convert stored messages back to LangChain message objects
        messages = []
        for msg_data in chat_data["messages"]:
            if msg_data["type"] == "HumanMessage":
                messages.append(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "AIMessage":
                messages.append(AIMessage(content=msg_data["content"]))
                
        return messages
        
    def get_chats_by_user(self, user_id: str) -> List[Dict]:
        """
        Get all chat sessions for a specific user.
        
        Args:
            user_id (str): The user's identifier
            
        Returns:
            List[Dict]: List of chat session data
        """
        chat_sessions = []
        
        for filename in os.listdir(self.storage_path):
            if not filename.endswith('.json'):
                continue
                
            with open(os.path.join(self.storage_path, filename), 'r') as f:
                chat_data = json.load(f)
                if chat_data["user_id"] == user_id:
                    chat_sessions.append(chat_data)
                    
        return chat_sessions
        
    def get_chats_by_procedure(self, procedure_id: str) -> List[Dict]:
        """
        Get all chat sessions for a specific procedure.
        
        Args:
            procedure_id (str): The procedure identifier
            
        Returns:
            List[Dict]: List of chat session data
        """
        chat_sessions = []
        
        for filename in os.listdir(self.storage_path):
            if not filename.endswith('.json'):
                continue
                
            with open(os.path.join(self.storage_path, filename), 'r') as f:
                chat_data = json.load(f)
                if chat_data["procedure_id"] == procedure_id:
                    chat_sessions.append(chat_data)
                    
        return chat_sessions