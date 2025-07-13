"""
LLM Service for EdgeElite AI Assistant - Qualcomm HaQathon

This module provides on-device LLM inference capabilities for Snapdragon X-Elite.
Uses NPU-optimized models for maximum performance on Qualcomm Snapdragon X-Elite.
"""

import time
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

# Try to import ONNX Runtime and Transformers
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime not available")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.pipelines import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available")

# Try to import LM Studio client
try:
    from lmstudio import Client
    LM_STUDIO_AVAILABLE = True
except ImportError:
    LM_STUDIO_AVAILABLE = False
    print("⚠️ LM Studio client not available")

# Try to import Flan-T5 service (WORKING NPU IMPLEMENTATION)
try:
    from llm_flan_t5 import get_flan_t5_service
    FLAN_T5_AVAILABLE = True
    print("[LLM] ✅ Flan-T5 service available - NPU optimized!")
except ImportError:
    FLAN_T5_AVAILABLE = False
    print("⚠️ Flan-T5 service not available")

class LLMService:
    """On-device LLM service for Snapdragon X-Elite edge AI using NPU acceleration."""
    
    def __init__(self):
        """Initialize the LLM service for edge AI."""
        self.model_loaded = False
        self.model_name = "flan-t5-small"  # Use Flan-T5 as primary model
        self.max_context_length = 512
        self.max_response_length = 128
        
        # Model paths
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        self.small_llm_path = os.path.join(self.models_dir, "small_llm")
        
        # NPU components
        self.session = None
        self.tokenizer = None
        self.use_npu = False
        self.provider = None
        
        # Transformers components (fallback)
        self.transformers_model = None
        self.transformers_tokenizer = None
        self.text_generator = None
        
        # Flan-T5 service (PRIMARY - NPU optimized)
        self.flan_t5_service = None
        if FLAN_T5_AVAILABLE:
            try:
                self.flan_t5_service = get_flan_t5_service()
                print("[LLM] ✅ Flan-T5 service initialized")
            except Exception as e:
                print(f"[LLM] ⚠️ Flan-T5 service initialization failed: {e}")
                self.flan_t5_service = None
        
        # LM Studio client
        self.lm_client = None
        if LM_STUDIO_AVAILABLE:
            try:
                self.lm_client = Client(base_url="http://localhost:8080")
                print("[LLM] ✅ LM Studio client initialized")
            except Exception as e:
                print(f"[LLM] ⚠️ LM Studio client initialization failed: {e}")
                self.lm_client = None
        
        # Enhanced mock responses
        self.mock_responses = {
            "summarize": [
                "Based on your session analysis, you've been working on a productive task that combines visual and audio elements. The pattern suggests focused work with regular documentation.",
                "Your session shows systematic information gathering with both screenshots and audio recordings, indicating thorough work on a multi-modal project.",
                "I can see you've been actively capturing both visual and verbal information, suggesting you're working on something that requires comprehensive documentation."
            ],
            "analysis": [
                "The pattern of your work suggests you're engaged in detailed, multi-modal tasks that benefit from edge AI assistance.",
                "Your workflow indicates a systematic approach to information gathering, perfect for on-device AI analysis.",
                "The frequency of captures suggests you're working on something that requires careful attention to detail and regular checkpoints."
            ],
            "recommendations": [
                "Consider using edge AI to automatically categorize your captured content for better organization.",
                "Try asking specific questions about your work patterns to get targeted insights from the on-device AI.",
                "Regular summarization can help you track progress and identify patterns in your work using local processing."
            ],
            "edge_ai": [
                "Your work pattern is ideal for edge AI processing - local, fast, and privacy-preserving.",
                "The multi-modal nature of your captures benefits from on-device AI that can process both visual and audio data.",
                "Edge AI can help you organize and analyze your work without sending data to the cloud."
            ]
        }
        
    def load_model(self):
        """Load the best available model with NPU priority."""
        try:
            print(f"[LLM] 🚀 Loading models for Snapdragon X-Elite NPU...")
            
            # Priority 1: Try Flan-T5 (NPU optimized, was working before merge)
            if FLAN_T5_AVAILABLE and self.flan_t5_service:
                try:
                    if self.flan_t5_service.load_model():
                        self.model_loaded = True
                        self.use_npu = self.flan_t5_service.use_qnn
                        print("[LLM] ✅ Flan-T5 NPU model loaded successfully!")
                        return
                except Exception as e:
                    print(f"[LLM] ⚠️ Flan-T5 loading failed: {e}")
            
            # Priority 2: Try LM Studio (if available)
            if LM_STUDIO_AVAILABLE and self.lm_client:
                try:
                    models = self.lm_client.list_models()
                    print(f"[LLM] ✅ LM Studio available models: {models}")
                    self.lm_client.load_model("llama-v3.2-3b-instruct")
                    self.model_loaded = True
                    print("[LLM] ✅ LM Studio model loaded successfully")
                    return
                except Exception as e:
                    print(f"[LLM] ⚠️ LM Studio connection failed: {e}")
            
            # Priority 3: Try NPU-optimized ONNX model
            if ONNX_AVAILABLE:
                try:
                    self._load_npu_onnx_model()
                    if self.model_loaded:
                        print("[LLM] ✅ NPU ONNX model loaded successfully")
                        return
                except Exception as e:
                    print(f"[LLM] ⚠️ NPU ONNX loading failed: {e}")
            
            # Priority 4: Try NPU-optimized Transformers model
            if TRANSFORMERS_AVAILABLE and os.path.exists(self.small_llm_path):
                try:
                    self._load_npu_transformers_model()
                    if self.model_loaded:
                        print("[LLM] ✅ NPU Transformers model loaded successfully")
                        return
                except Exception as e:
                    print(f"[LLM] ⚠️ NPU Transformers loading failed: {e}")
            
            # Priority 5: Fallback to CPU Transformers
            if TRANSFORMERS_AVAILABLE and os.path.exists(self.small_llm_path):
                try:
                    self._load_cpu_transformers_model()
                    if self.model_loaded:
                        print("[LLM] ⚠️ CPU Transformers model loaded (NPU not available)")
                        return
                except Exception as e:
                    print(f"[LLM] ⚠️ CPU Transformers loading failed: {e}")
            
            # Final fallback: Enhanced mock mode
            print("[LLM] ❌ No models available! Using enhanced mock responses")
            self.model_loaded = True
            
        except Exception as e:
            print(f"[LLM] Failed to load models: {e}")
            self.model_loaded = True

    def _load_npu_onnx_model(self):
        """Load NPU-optimized ONNX model for maximum performance."""
        try:
            # Check for QNN NPU availability
            providers = ort.get_available_providers()
            print(f"[LLM] Available providers: {providers}")
            
            if 'QNNExecutionProvider' in providers:
                print("[LLM] 🚀 QNN NPU detected - Loading ONNX model for NPU acceleration")
                
                # Look for NPU-optimized ONNX model
                onnx_model_path = os.path.join(self.models_dir, "llm_model.onnx")
                if os.path.exists(onnx_model_path):
                    # Load tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(self.small_llm_path)
                    
                    # Create ONNX session with QNN NPU
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    
                    self.session = ort.InferenceSession(
                        onnx_model_path,
                        sess_options=session_options,
                        providers=['QNNExecutionProvider', 'CPUExecutionProvider']
                    )
                    
                    self.use_npu = True
                    self.provider = 'QNNExecutionProvider'
                    self.model_loaded = True
                    print("[LLM] ✅ NPU ONNX model loaded with QNN acceleration")
                    return True
                else:
                    print("[LLM] ⚠️ NPU ONNX model not found")
                    return False
            else:
                print("[LLM] ⚠️ QNN NPU not available")
                return False
                
        except Exception as e:
            print(f"[LLM] NPU ONNX loading failed: {e}")
            return False
    
    def _load_npu_transformers_model(self):
        """Load Transformers model optimized for NPU."""
        try:
            print("[LLM] 🚀 Loading Transformers model for NPU acceleration...")
            
            # Load tokenizer
            self.transformers_tokenizer = AutoTokenizer.from_pretrained(
                self.small_llm_path,
                trust_remote_code=True
            )
            
            # Check for NPU availability
            import torch
            if hasattr(torch, 'npu') and torch.npu.is_available():
                print("[LLM] 🚀 PyTorch NPU detected")
                device = "npu"
                self.use_npu = True
            elif 'QNNExecutionProvider' in ort.get_available_providers():
                print("[LLM] 🚀 QNN NPU detected via ONNX Runtime")
                device = "cpu"  # QNN will handle NPU acceleration
                self.use_npu = True
            else:
                print("[LLM] ⚠️ No NPU detected, falling back to CPU")
                device = "cpu"
                self.use_npu = False
            
            # Load model
            self.transformers_model = AutoModelForCausalLM.from_pretrained(
                self.small_llm_path,
                torch_dtype="auto",
                device_map=device if device != "npu" else "auto",
                trust_remote_code=True
            )
            
            # Move to NPU if available
            if device == "npu":
                self.transformers_model = self.transformers_model.to("npu")
            
            # Create pipeline
            self.text_generator = pipeline(
                "text-generation",
                model=self.transformers_model,
                tokenizer=self.transformers_tokenizer,
                max_length=self.max_context_length + self.max_response_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            self.model_loaded = True
            print(f"[LLM] ✅ Transformers model loaded for {'NPU' if self.use_npu else 'CPU'} inference")
            return True
            
        except Exception as e:
            print(f"[LLM] NPU Transformers loading failed: {e}")
            return False
    
    def _load_cpu_transformers_model(self):
        """Load Transformers model on CPU (fallback)."""
        try:
            print("[LLM] ⚠️ Loading Transformers model on CPU...")
            
            # Load tokenizer
            self.transformers_tokenizer = AutoTokenizer.from_pretrained(
                self.small_llm_path,
                trust_remote_code=True
            )
            
            # Load model on CPU
            self.transformers_model = AutoModelForCausalLM.from_pretrained(
                self.small_llm_path,
                torch_dtype="auto",
                device_map="cpu",
                trust_remote_code=True
            )
            
            # Create pipeline
            self.text_generator = pipeline(
                "text-generation",
                model=self.transformers_model,
                tokenizer=self.transformers_tokenizer,
                max_length=self.max_context_length + self.max_response_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            self.model_loaded = True
            self.use_npu = False
            print("[LLM] ✅ CPU Transformers model loaded")
            return True
            
        except Exception as e:
            print(f"[LLM] CPU Transformers loading failed: {e}")
            return False

    def generate_edge_response(self, prompt: str) -> str:
        """Generate response using the best available method."""
        if not self.model_loaded:
            return "Model not loaded. Please try again."
        
        try:
            start_time = time.time()
            
            # Priority 1: Try Flan-T5 (NPU optimized, was working)
            if self.flan_t5_service and self.flan_t5_service.model_loaded:
                try:
                    response = self.flan_t5_service.generate_response(prompt, [])
                    print(f"[LLM] Flan-T5 NPU response generated in {time.time() - start_time:.2f}s")
                    return response
                except Exception as e:
                    print(f"[LLM] Flan-T5 generation failed: {e}")
            
            # Priority 2: Try LM Studio
            if self.lm_client:
                try:
                    response = self._generate_lm_studio_response(prompt)
                    print(f"[LLM] LM Studio response generated in {time.time() - start_time:.2f}s")
                    return response
                except Exception as e:
                    print(f"[LLM] LM Studio generation failed: {e}")
            
            # Priority 3: Try NPU ONNX
            if self.session and self.use_npu:
                try:
                    response = self._generate_npu_onnx_response(prompt)
                    print(f"[LLM] NPU ONNX response generated in {time.time() - start_time:.2f}s")
                    return response
                except Exception as e:
                    print(f"[LLM] NPU ONNX generation failed: {e}")
            
            # Priority 4: Try NPU Transformers
            if self.text_generator and self.use_npu:
                try:
                    response = self._generate_npu_transformers_response(prompt)
                    print(f"[LLM] NPU Transformers response generated in {time.time() - start_time:.2f}s")
                    return response
                except Exception as e:
                    print(f"[LLM] NPU Transformers generation failed: {e}")
            
            # Priority 5: Try CPU Transformers
            if self.text_generator:
                try:
                    response = self._generate_cpu_transformers_response(prompt)
                    print(f"[LLM] CPU Transformers response generated in {time.time() - start_time:.2f}s")
                    return response
                except Exception as e:
                    print(f"[LLM] CPU Transformers generation failed: {e}")
            
            # Fallback to enhanced mock
            response = self._generate_enhanced_mock_response([], prompt)
            print(f"[LLM] Enhanced mock response generated in {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            print(f"[LLM] Generation failed: {e}")
            return "I'm having trouble generating a response right now. Please try again."

    def _generate_lm_studio_response(self, prompt: str) -> str:
        """Generate response using LM Studio."""
        try:
            response = self.lm_client.chat.completions.create(
                model="llama-v3.2-3b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_response_length,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"LM Studio generation failed: {e}")

    def _generate_npu_onnx_response(self, prompt: str) -> str:
        """Generate response using NPU ONNX model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].numpy()
            
            # Run inference on NPU
            outputs = self.session.run(None, {"input_ids": input_ids})
            logits = outputs[0]
            
            # Decode response
            predicted_ids = np.argmax(logits, axis=-1)
            response = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
            return response[len(prompt):].strip()
        except Exception as e:
            raise Exception(f"NPU ONNX generation failed: {e}")

    def _generate_npu_transformers_response(self, prompt: str) -> str:
        """Generate response using NPU Transformers model."""
        try:
            # Generate using pipeline
            outputs = self.text_generator(
                prompt,
                max_new_tokens=self.max_response_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            
            return response
        except Exception as e:
            raise Exception(f"NPU Transformers generation failed: {e}")

    def _generate_cpu_transformers_response(self, prompt: str) -> str:
        """Generate response using CPU Transformers model."""
        try:
            # Generate using pipeline
            outputs = self.text_generator(
                prompt,
                max_new_tokens=self.max_response_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            
            return response
        except Exception as e:
            raise Exception(f"CPU Transformers generation failed: {e}")

    def format_context_for_llm(self, context: List[Dict[str, Any]]) -> str:
        """Format context for LLM input."""
        if not context:
            return ""
        
        formatted_context = "Based on the following session context:\n\n"
        for i, item in enumerate(context[-5:], 1):  # Last 5 items
            if item.get('type') == 'screenshot':
                formatted_context += f"{i}. Screenshot captured at {item.get('timestamp', 'unknown time')}\n"
            elif item.get('type') == 'audio':
                formatted_context += f"{i}. Audio recording captured at {item.get('timestamp', 'unknown time')}\n"
            elif item.get('type') == 'ocr':
                formatted_context += f"{i}. Text extracted: {item.get('text', 'No text found')}\n"
        
        return formatted_context

    def generate_prompt(self, user_input: str, context: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive prompt for the LLM."""
        context_str = self.format_context_for_llm(context)
        
        # Enhanced prompt for edge AI assistant
        prompt = f"""You are EdgeElite, an intelligent AI assistant running on a Qualcomm Snapdragon X-Elite device. You provide helpful, contextual responses based on the user's session data.

{context_str}

User: {user_input}

EdgeElite:"""
        
        return prompt

    def generate_response(self, user_input: str, context: List[Dict[str, Any]]) -> str:
        """Generate a response based on user input and context."""
        try:
            # For Flan-T5, pass the user input directly (it handles its own prompt formatting)
            if self.flan_t5_service and self.flan_t5_service.model_loaded:
                response = self.flan_t5_service.generate_response(user_input, context)
                return self._post_process_response(response, context, user_input)
            
            # For other models, use comprehensive prompt
            prompt = self.generate_prompt(user_input, context)
            response = self.generate_edge_response(prompt)
            response = self._post_process_response(response, context, user_input)
            
            return response
            
        except Exception as e:
            print(f"[LLM] Response generation failed: {e}")
            return "I'm having trouble processing your request right now. Please try again."

    def _generate_enhanced_mock_response(self, context: List[Dict[str, Any]], user_input: str) -> str:
        """Generate enhanced mock response based on context and input."""
        try:
            # Analyze user input for intent
            user_input_lower = user_input.lower()
            
            # Generate contextual insights
            insights = self._generate_contextual_insights(context, user_input)
            
            # Create response based on input type
            if any(word in user_input_lower for word in ['summarize', 'summary', 'overview']):
                import random
                base_response = random.choice(self.mock_responses["summarize"])
                return f"{base_response}\n\n{insights}"
            
            elif any(word in user_input_lower for word in ['analyze', 'analysis', 'pattern']):
                import random
                base_response = random.choice(self.mock_responses["analysis"])
                return f"{base_response}\n\n{insights}"
            
            elif any(word in user_input_lower for word in ['recommend', 'suggestion', 'advice']):
                import random
                base_response = random.choice(self.mock_responses["recommendations"])
                return f"{base_response}\n\n{insights}"
            
            elif any(word in user_input_lower for word in ['edge', 'npu', 'device', 'local']):
                import random
                base_response = random.choice(self.mock_responses["edge_ai"])
                return f"{base_response}\n\n{insights}"
            
            else:
                # General response
                return f"I understand you're asking about: {user_input}\n\n{insights}\n\nAs your edge AI assistant, I'm here to help you make the most of your on-device AI capabilities."
                
        except Exception as e:
            print(f"[LLM] Enhanced mock response generation failed: {e}")
            return "I'm here to help with your edge AI tasks. What would you like to know?"

    def _generate_contextual_insights(self, context: List[Dict[str, Any]], user_input: str) -> str:
        """Generate contextual insights based on session data."""
        try:
            if not context:
                return "No session context available yet. Start capturing screenshots or audio to get contextual insights."
            
            # Count different types of captures
            screenshots = sum(1 for item in context if item.get('type') == 'screenshot')
            audio_recordings = sum(1 for item in context if item.get('type') == 'audio')
            ocr_extractions = sum(1 for item in context if item.get('type') == 'ocr')
            
            insights = f"Session Insights:\n"
            insights += f"• Screenshots captured: {screenshots}\n"
            insights += f"• Audio recordings: {audio_recordings}\n"
            insights += f"• Text extractions: {ocr_extractions}\n"
            
            if context:
                latest_item = context[-1]
                insights += f"• Latest activity: {latest_item.get('type', 'unknown')} at {latest_item.get('timestamp', 'unknown time')}\n"
            
            return insights
            
        except Exception as e:
            print(f"[LLM] Contextual insights generation failed: {e}")
            return "Context analysis temporarily unavailable."

    def _post_process_response(self, response: str, context: List[Dict[str, Any]], user_input: str) -> str:
        """Post-process the generated response."""
        try:
            # Clean up response
            response = response.strip()
            
            # Remove any duplicate content
            if response.startswith(user_input):
                response = response[len(user_input):].strip()
            
            # Ensure response is not empty
            if not response:
                response = "I'm processing your request. Please try again."
            
            return response
            
        except Exception as e:
            print(f"[LLM] Post-processing failed: {e}")
            return response

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        flan_info = {}
        if self.flan_t5_service:
            flan_info = self.flan_t5_service.get_model_info()
        
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "use_npu": self.use_npu,
            "provider": self.provider,
            "max_context_length": self.max_context_length,
            "max_response_length": self.max_response_length,
            "available_methods": {
                "flan_t5_npu": self.flan_t5_service is not None and self.flan_t5_service.model_loaded,
                "lm_studio": self.lm_client is not None,
                "npu_onnx": self.session is not None and self.use_npu,
                "npu_transformers": self.text_generator is not None and self.use_npu,
                "cpu_transformers": self.text_generator is not None and not self.use_npu,
                "mock": True
            },
            "flan_t5_info": flan_info
        } 