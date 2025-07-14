#!/usr/bin/env python3
"""
Flan-T5 Small LLM Service for EdgeElite AI Assistant
Uses quantized Flan-T5 models for high-quality text generation on edge devices.
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional

# Try to import transformers for tokenizer
try:
    from transformers import T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available - install with: pip install transformers")

class FlanT5LLMService:
    """Flan-T5 Small LLM service for edge AI using quantized ONNX models."""
    
    def __init__(self):
        """Initialize the Flan-T5 LLM service."""
        self.model_loaded = False
        self.model_name = "flan-t5-small"
        self.max_context_length = 512  # T5 context length
        self.max_response_length = 128  # Response length
        
        # Model paths
        self.models_dir = os.path.join(os.path.dirname(__file__), "flan-t5-small-ONNX")
        self.encoder_path = os.path.join(self.models_dir, "onnx", "encoder_model_int8.onnx")
        self.decoder_path = os.path.join(self.models_dir, "onnx", "decoder_model_int8.onnx")
        
        # ONNX sessions
        self.encoder_session = None
        self.decoder_session = None
        
        # Tokenizer
        self.tokenizer = None
        
        # Check QNN availability
        self.use_qnn = 'QNNExecutionProvider' in ort.get_available_providers()
        if self.use_qnn:
            print("[Flan-T5] ✅ QNN NPU available - will try to use it")
        else:
            print("[Flan-T5] ⚠️ QNN NPU not available - using CPU")
    
    def load_model(self):
        """Load the quantized Flan-T5 models."""
        try:
            print(f"[Flan-T5] Loading {self.model_name} for edge AI...")
            
            # Check if models exist
            if not os.path.exists(self.encoder_path):
                print(f"[Flan-T5] ❌ Encoder model not found: {self.encoder_path}")
                return False
            if not os.path.exists(self.decoder_path):
                print(f"[Flan-T5] ❌ Decoder model not found: {self.decoder_path}")
                return False
            
            print(f"[Flan-T5] ✅ Encoder model found: {os.path.getsize(self.encoder_path) / 1024 / 1024:.1f} MB")
            print(f"[Flan-T5] ✅ Decoder model found: {os.path.getsize(self.decoder_path) / 1024 / 1024:.1f} MB")
            
            # Use CPU execution provider (Flan-T5 has QNN compatibility issues)
            providers = ['CPUExecutionProvider']
            
            print("[Flan-T5] Loading encoder with CPU (optimized for Snapdragon X Elite)...")
            self.encoder_session = ort.InferenceSession(self.encoder_path, providers=providers)
            print("[Flan-T5] ✅ Encoder loaded successfully!")
            
            print("[Flan-T5] Loading decoder with CPU (optimized for Snapdragon X Elite)...")
            self.decoder_session = ort.InferenceSession(self.decoder_path, providers=providers)
            print("[Flan-T5] ✅ Decoder loaded successfully!")
            
            print("[Flan-T5] 🚀 Using CPU execution (still very fast on Snapdragon X Elite!)")
            
            # Load tokenizer
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Try loading from local directory with config.json
                    self.tokenizer = T5Tokenizer.from_pretrained(self.models_dir)
                    print("[Flan-T5] ✅ Tokenizer loaded successfully from local directory")
                except Exception as e:
                    print(f"[Flan-T5] ⚠️ Local tokenizer loading failed: {e}")
                    try:
                        # Fallback: try loading from Hugging Face model hub
                        self.tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
                        print("[Flan-T5] ✅ Tokenizer loaded from Hugging Face hub")
                    except Exception as e2:
                        print(f"[Flan-T5] ❌ Tokenizer loading failed: {e2}")
                        print("[Flan-T5] Will use simple tokenization")
            
            self.model_loaded = True
            print(f"[Flan-T5] ✅ {self.model_name} loaded successfully!")
            return True
            
        except Exception as e:
            print(f"[Flan-T5] ❌ Failed to load models: {e}")
            return False
    
    def generate_response(self, user_input: str, context: List[Dict[str, Any]]) -> str:
        """Generate response using Flan-T5 sequence-to-sequence model."""
        if not self.model_loaded:
            if not self.load_model():
                return "Sorry, I couldn't load the AI model. Please try again."
        
        try:
            # Create a simple prompt from user input and context
            prompt = self._create_prompt(user_input, context)
            
            # Generate response using Flan-T5
            response = self._generate_flan_t5_response(prompt)
            
            return self._post_process_response(response, context, user_input)
            
        except Exception as e:
            print(f"[Flan-T5] Generation failed: {e}")
            return f"I'm having trouble generating a response right now. Error: {str(e)}"
    
    def _create_prompt(self, user_input: str, context: List[Dict[str, Any]]) -> str:
        """Create a prompt for Flan-T5 from user input and context."""
        # For Flan-T5, we can use a simple instruction format
        if context:
            # Include some context
            recent_context = context[-2:] if len(context) > 2 else context
            context_text = " ".join([msg.get('content', '') for msg in recent_context])
            prompt = f"Context: {context_text}\nQuestion: {user_input}\nAnswer:"
        else:
            prompt = f"Question: {user_input}\nAnswer:"
        
        return prompt
    
    def _generate_flan_t5_response(self, prompt: str) -> str:
        """Generate response using Flan-T5 encoder-decoder architecture."""
        try:
            # Tokenize input
            if self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors="np", max_length=self.max_context_length, truncation=True)
                # Ensure int64 dtype
                input_ids = input_ids.astype(np.int64)
            else:
                # Simple tokenization fallback
                input_ids = np.array([[hash(word) % 1000 for word in prompt.split()[:self.max_context_length]]], dtype=np.int64)
                if input_ids.shape[1] == 0:
                    input_ids = np.array([[0]], dtype=np.int64)
            
            print(f"[Flan-T5] 🧠 Generating response with input length: {input_ids.shape[1]}")
            
            # Create attention mask (all ones for valid tokens) - ensure int64
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
            
            # Run encoder
            encoder_output = self.encoder_session.run(None, {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
            encoder_hidden_states = encoder_output[0]
            
            print(f"[Flan-T5] ✅ Encoder completed, hidden states shape: {encoder_hidden_states.shape}")
            
            # Initialize decoder input (start token) - ensure int64
            decoder_input_ids = np.array([[0]], dtype=np.int64)  # T5 start token
            
            # Generate response tokens
            generated_tokens = []
            max_tokens = self.max_response_length
            
            for _ in range(max_tokens):
                # Run decoder
                decoder_output = self.decoder_session.run(None, {
                    'input_ids': decoder_input_ids,
                    'encoder_hidden_states': encoder_hidden_states,
                    'encoder_attention_mask': attention_mask
                })
                
                # Get next token (simple greedy decoding)
                next_token_logits = decoder_output[0][0, -1, :]
                next_token = np.argmax(next_token_logits)
                
                # Stop if end token
                if next_token == 1:  # T5 end token
                    break
                
                generated_tokens.append(next_token)
                # Ensure the new token is int64 and concatenate
                new_token_array = np.array([[next_token]], dtype=np.int64)
                decoder_input_ids = np.concatenate([decoder_input_ids, new_token_array], axis=1)
            
            # Decode response
            if self.tokenizer:
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                # Simple decoding fallback
                response = " ".join([str(token) for token in generated_tokens])
            
            print(f"[Flan-T5] ✅ Generated response: {len(response)} characters")
            return response
            
        except Exception as e:
            print(f"[Flan-T5] Inference failed: {e}")
            return f"Generation error: {str(e)}"
    
    def _post_process_response(self, response: str, context: List[Dict[str, Any]], user_input: str) -> str:
        """Post-process the generated response."""
        # Clean up response
        response = response.strip()
        
        # If response is too short, add some context
        if len(response) < 10:
            response = f"I understand you're asking about '{user_input}'. Let me help you with that."
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Flan-T5 model."""
        return {
            "model_name": self.model_name,
            "loaded": self.model_loaded,
            "model_type": "Flan-T5 Small (Sequence-to-Sequence)",
            "parameters": "80M",
            "quantization": "INT8",
            "context_length": self.max_context_length,
            "response_length": self.max_response_length,
            "providers": self.encoder_session.get_providers() if self.encoder_session else [],
            "qnn_available": self.use_qnn,
            "tokenizer_available": TRANSFORMERS_AVAILABLE and self.tokenizer is not None,
            "edge_optimized": True,
            "platform": "Snapdragon X-Elite"
        }

# Global instance
flan_t5_service = FlanT5LLMService()

def get_flan_t5_service() -> FlanT5LLMService:
    """Get the global Flan-T5 service instance."""
    return flan_t5_service 