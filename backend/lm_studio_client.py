#!/usr/bin/env python3
"""
LM Studio Client for EdgeElite AI Assistant
Connect to LM Studio to run models remotely without downloading them locally.

LM Studio provides a local API server that can run various models.
This client connects to LM Studio's API for real AI responses.
"""

import requests
import json
import time
from typing import Dict, Any, Optional, List
import logging

class LMStudioClient:
    """Client for connecting to LM Studio API server."""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        """Initialize LM Studio client.
        
        Args:
            base_url: LM Studio API server URL (default: http://localhost:1234)
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/v1"
        self.chat_url = f"{self.api_url}/chat/completions"
        self.completions_url = f"{self.api_url}/completions"
        self.models_url = f"{self.api_url}/models"
        
        # Connection status
        self.connected = False
        self.available_models = []
        
        # Default settings
        self.default_model = "llama-v3.2-3b-instruct"  # Use Llama v3.2 3B Instruct
        self.max_tokens = 512
        self.temperature = 0.7
        self.top_p = 0.95
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to LM Studio server."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                self.connected = True
                print(f"✅ Connected to LM Studio at {self.base_url}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to LM Studio: {e}")
            print(f"   Make sure LM Studio is running and the API server is enabled")
            print(f"   Expected URL: {self.base_url}")
        
        self.connected = False
        return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from LM Studio."""
        if not self.connected:
            return []
        
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = models_data.get("data", [])
                return self.available_models
        except Exception as e:
            print(f"Failed to get models: {e}")
        
        return []
    
    def generate_chat_response(self, messages: List[Dict[str, str]], 
                             model: Optional[str] = None,
                             max_tokens: Optional[int] = None,
                             temperature: Optional[float] = None) -> str:
        """Generate chat response using LM Studio.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name to use (default: self.default_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.connected:
            raise ConnectionError("Not connected to LM Studio")
        
        # Prepare request payload
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": self.top_p,
            "stream": False
        }
        
        try:
            print(f"[LM Studio] Generating response with model: {payload['model']}")
            response = requests.post(self.chat_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"[LM Studio] ✅ Response generated successfully")
                return content
            else:
                error_msg = f"LM Studio API error: {response.status_code} - {response.text}"
                print(f"[LM Studio] ❌ {error_msg}")
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "LM Studio request timed out"
            print(f"[LM Studio] ❌ {error_msg}")
            raise Exception(error_msg)
        except Exception as e:
            print(f"[LM Studio] ❌ Request failed: {e}")
            raise
    
    def generate_completion(self, prompt: str,
                          model: Optional[str] = None,
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None) -> str:
        """Generate completion using LM Studio.
        
        Args:
            prompt: Input prompt text
            model: Model name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated completion text
        """
        if not self.connected:
            raise ConnectionError("Not connected to LM Studio")
        
        # Prepare request payload
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": self.top_p,
            "stream": False
        }
        
        try:
            print(f"[LM Studio] Generating completion with model: {payload['model']}")
            response = requests.post(self.completions_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["text"]
                print(f"[LM Studio] ✅ Completion generated successfully")
                return text
            else:
                error_msg = f"LM Studio API error: {response.status_code} - {response.text}"
                print(f"[LM Studio] ❌ {error_msg}")
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "LM Studio request timed out"
            print(f"[LM Studio] ❌ {error_msg}")
            raise Exception(error_msg)
        except Exception as e:
            print(f"[LM Studio] ❌ Request failed: {e}")
            raise
    
    def format_context_for_lm_studio(self, context: List[Dict[str, Any]]) -> str:
        """Format context for LM Studio input."""
        if not context:
            return "No context available."
        
        formatted_context = []
        for i, event in enumerate(context, 1):
            source = event.get('source', 'unknown')
            text = event.get('text', '')
            timestamp = event.get('metadata', {}).get('timestamp', '')
            
            formatted_context.append(f"{i}. [{source.upper()}] {timestamp}: {text}")
        
        return "\n".join(formatted_context)
    
    def generate_edge_response(self, user_input: str, context: List[Dict[str, Any]]) -> str:
        """Generate response for EdgeElite using LM Studio."""
        if not self.connected:
            raise ConnectionError("LM Studio not connected")
        
        # Format context
        context_text = self.format_context_for_lm_studio(context)
        
        # Create system message
        system_message = """You are an edge AI assistant running on Snapdragon X-Elite, analyzing user sessions and providing helpful insights. 
You have access to context from the user's work session including screenshots and audio recordings.
Provide concise, practical responses with insights and recommendations."""
        
        # Create user message
        user_message = f"""Context from user's session:
{context_text}

User Query: {user_input}

Please provide a helpful response with insights and recommendations. Be concise and practical."""
        
        # Prepare messages for chat API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Generate response
        return self.generate_chat_response(messages)
    
    def get_status(self) -> Dict[str, Any]:
        """Get LM Studio connection status and information."""
        return {
            "connected": self.connected,
            "base_url": self.base_url,
            "available_models": len(self.available_models),
            "default_model": self.default_model,
            "models": [model.get("id", "unknown") for model in self.available_models[:5]]  # Show first 5
        }

# Global instance
lm_studio_client = LMStudioClient() 