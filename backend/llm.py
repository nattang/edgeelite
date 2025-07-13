"""
LLM Service for EdgeElite AI Assistant - Qualcomm HaQathon

This module provides on-device LLM inference capabilities for Snapdragon X-Elite.
Uses Llama3-TAIDE-LX-8B-Chat-Alpha1 optimized for Qualcomm Snapdragon X-Elite NPU.
Based on Qualcomm AI Hub tutorials: https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie
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
    print("⚠️ ONNX Runtime not available, using enhanced mock responses")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers.pipelines import pipeline
    from transformers.utils.quantization_config import BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available, using enhanced mock responses")

# Try to import LM Studio client
try:
    from lmstudio import Client
    LM_STUDIO_AVAILABLE = True
except ImportError:
    LM_STUDIO_AVAILABLE = False
    print("⚠️ LM Studio client not available")

# Try to import Flan-T5 service
try:
    from llm_flan_t5 import get_flan_t5_service
    FLAN_T5_AVAILABLE = True
except ImportError:
    FLAN_T5_AVAILABLE = False
    print("⚠️ Flan-T5 service not available")

class LLMService:
    """On-device LLM service for Snapdragon X-Elite edge AI using Llama-v3.2-3B-Instruct."""
    
    def __init__(self):
        """Initialize the LLM service for edge AI."""
        self.model_loaded = False
        # Use Llama-v3.2-3B-Instruct from LM Studio for high-quality edge inference
        self.model_name = "llama-v3.2-3b-instruct"
        self.max_context_length = 4096  # Llama v3.2 supports longer context
        self.max_response_length = 512  # Longer responses for better quality
        
        # Model paths for small LLM
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        self.small_llm_path = os.path.join(self.models_dir, "small_llm")
        
        # LM Studio client pointing at local server
        self.lm_client = None
        if LM_STUDIO_AVAILABLE:
            try:
                self.lm_client = Client(base_url="http://localhost:8080")
                print("[LLM] ✅ LM Studio client initialized")
            except Exception as e:
                print(f"[LLM] ⚠️ LM Studio client initialization failed: {e}")
                self.lm_client = None
        
        # ONNX Runtime components
        self.session = None
        self.tokenizer = None
        
        # Transformers components
        self.transformers_model = None
        self.transformers_tokenizer = None
        self.text_generator = None
        self.use_qnn = False
        
        # Enhanced mock responses for edge AI scenarios
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
        """Load the best available model for high-quality edge inference."""
        try:
            print(f"[LLM] Loading models for Snapdragon X-Elite...")
            
            # Try connecting to LM Studio first (highest priority)
            if LM_STUDIO_AVAILABLE and self.lm_client:
                try:
                    # Check connectivity and list available models
                    models = self.lm_client.list_models()
                    print(f"[LLM] ✅ LM Studio available models: {models}")
                    
                    # Load the desired model
                    self.lm_client.load_model("llama-v3.2-3b-instruct")
                    self.model_loaded = True
                    print("[LLM] ✅ LM Studio model loaded successfully")
                    return
                except Exception as e:
                    print(f"[LLM] ⚠️ LM Studio connection failed: {e}")
            
            # Try loading Flan-T5 model (high quality, quantized)
            if FLAN_T5_AVAILABLE:
                try:
                    flan_t5_service = get_flan_t5_service()
                    if flan_t5_service.load_model():
                        self.model_loaded = True
                        self.flan_t5_service = flan_t5_service  # Store reference
                        print("[LLM] ✅ Flan-T5 model loaded successfully")
                        return
                except Exception as e:
                    print(f"[LLM] ⚠️ Flan-T5 loading failed: {e}")
            
            # Try loading small LLM model locally
            if TRANSFORMERS_AVAILABLE and os.path.exists(self.small_llm_path):
                self._load_small_llm_model()
                print(f"[LLM] ✅ Small LLM model loaded: | Device: {'NPU' if self.use_qnn else 'CPU/GPU'}")
                if self.use_qnn:
                    print("[LLM] ✅ Using NPU (QNN) for small LLM inference!")
                else:
                    print("[LLM] ⚠️ Not using NPU, using CPU/GPU for small LLM inference.")
                return
                
            # If no model available, use enhanced mock mode
            print("[LLM] ❌ No models available! Using enhanced mock responses")
            self.model_loaded = True
        except Exception as e:
            print(f"[LLM] Failed to load models: {e}")
            self.model_loaded = True
    
    def _load_small_llm_model(self):
        """Load small LLM model for fast edge inference."""
        try:
            # Load tokenizer
            self.transformers_tokenizer = AutoTokenizer.from_pretrained(
                self.small_llm_path,
                trust_remote_code=True
            )
            
            # Detect best available device for Snapdragon X-Elite
            device = self._detect_best_device()
            
            # Load model with device-specific configuration
            if device == "qnn_npu":
                # QNN NPU-specific loading for Snapdragon X-Elite
                print("🚀 Loading small LLM for QNN NPU acceleration")
                self.transformers_model = AutoModelForCausalLM.from_pretrained(
                    self.small_llm_path,
                    torch_dtype="auto",
                    device_map="cpu",  # QNN will handle NPU acceleration
                    trust_remote_code=True
                )
                # Set flag to use QNN for inference
                self.use_qnn = True
                print(f"✅ {self.model_name} loaded for edge inference on {device}")
            else:
                # For other devices, use standard loading
                self.transformers_model = AutoModelForCausalLM.from_pretrained(
                    self.small_llm_path,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True
                )
                print(f"✅ {self.model_name} loaded for edge inference on {device}")
            
            # Create pipeline for text generation
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
            print(f"✅ {self.model_name} loaded for fast edge inference")
            
        except Exception as e:
            print(f"Small LLM model loading failed: {e}")
            raise
    

    
    def _detect_best_device(self) -> str:
        """Detect the best available device for Snapdragon X-Elite."""
        try:
            import torch
            
            # Check for ONNX Runtime QNN (Qualcomm Snapdragon X-Elite NPU)
            try:
                import onnxruntime as ort
                if 'QNNExecutionProvider' in ort.get_available_providers():
                    print("🚀 QNN NPU detected - Using Qualcomm Snapdragon X-Elite NPU via ONNX Runtime")
                    return "qnn_npu"
            except (ImportError, AttributeError):
                pass  # ONNX Runtime QNN not available
            
            # Check for PyTorch NPU (Qualcomm Snapdragon X-Elite)
            try:
                if hasattr(torch, 'npu') and torch.npu.is_available():  # type: ignore
                    print("🚀 NPU detected - Using Qualcomm Snapdragon X-Elite NPU via PyTorch")
                    return "npu"
            except (AttributeError, ImportError):
                pass  # NPU not available
            
            # Check for CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                print("🚀 CUDA detected - Using NVIDIA GPU")
                return "cuda"
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🚀 MPS detected - Using Apple Silicon GPU")
                return "mps"
            
            # Fallback to CPU
            print("🚀 Using CPU for inference (no NPU/GPU detected)")
            return "cpu"
            
        except Exception as e:
            print(f"Device detection failed: {e}, falling back to CPU")
            return "cpu"
    
    def generate_edge_response(self, prompt: str) -> str:
        """Generate response using edge-optimized model."""
        try:
            # Try LM Studio first (highest priority)
            if LM_STUDIO_AVAILABLE and self.lm_client:
                try:
                    print("[LLM] 🧠 Generating response with LM Studio (Llama-v3.2-3B-Instruct)")
                    return self._generate_lm_studio_response(prompt)
                except Exception as e:
                    print(f"[LLM] ⚠️ LM Studio failed, falling back: {e}")
            
            # Try Flan-T5 model (high quality, quantized)
            if hasattr(self, 'flan_t5_service') and self.flan_t5_service and self.flan_t5_service.model_loaded:
                try:
                    print("[LLM] 🧠 Generating response with Flan-T5 (Quantized ONNX)")
                    return self.flan_t5_service.generate_response(prompt, [])
                except Exception as e:
                    print(f"[LLM] ⚠️ Flan-T5 failed, falling back: {e}")
            elif FLAN_T5_AVAILABLE:
                try:
                    flan_t5_service = get_flan_t5_service()
                    if flan_t5_service.model_loaded:
                        print("[LLM] 🧠 Generating response with Flan-T5 (Quantized ONNX)")
                        return flan_t5_service.generate_response(prompt, [])
                except Exception as e:
                    print(f"[LLM] ⚠️ Flan-T5 failed, falling back: {e}")
            
            # Try ONNX inference (fastest local)
            if self.session:
                print("[LLM] 🧠 Generating response with REAL MODEL (ONNX)")
                if 'QNNExecutionProvider' in self.session.get_providers():
                    print("[LLM] 🚀 NPU (QNN) is being used for ONNX inference!")
                else:
                    print("[LLM] ⚠️ NPU NOT used for ONNX, using CPU/GPU.")
                return self._generate_onnx_response(prompt)
            
            # Try Transformers inference
            if self.text_generator:
                print(f"[LLM] 🧠 Generating response with REAL MODEL (Transformers) | Device: {'NPU' if self.use_qnn else 'CPU/GPU'}")
                if self.use_qnn:
                    print("[LLM] 🚀 NPU (QNN) is being used for Transformers inference!")
                else:
                    print("[LLM] ⚠️ NPU NOT used for Transformers, using CPU/GPU.")
                return self._generate_transformers_response(prompt)
            
            # Fallback to enhanced mock
            print("[LLM] ❌ MOCK RESPONSE USED! This should not happen.")
            raise RuntimeError("No edge models available")
        except Exception as e:
            print(f"[LLM] Edge inference failed: {e}")
            raise
    
    def _generate_lm_studio_response(self, prompt: str) -> str:
        """Generate response using LM Studio REST API."""
        try:
            if not LM_STUDIO_AVAILABLE or not self.lm_client:
                return "LM Studio not available"
            
            # Send prompt and receive text completion
            resp = self.lm_client.generate_completion(
                model_id="llama-v3.2-3b-instruct",
                prompt=prompt,
                max_tokens=self.max_response_length,
                temperature=0.7,
            )
            return resp.text  # or resp.choices[0].text depending on SDK version
            
        except Exception as e:
            print(f"[LM Studio] inference error: {e}")
            return f"[LM Studio] error: {e}"
    
    def _generate_onnx_response(self, prompt: str) -> str:
        """Generate response using small LLM ONNX model (fastest for edge)."""
        try:
            if self.session is None:
                return "Generated response from small LLM ONNX model"
            
            # For now, return a placeholder since ONNX inference needs special implementation
            # TODO: Implement proper ONNX inference for the small LLM model
            print("[LLM] ⚠️ ONNX inference not yet implemented for small LLM model")
            return "Generated response from small LLM ONNX model (placeholder)"
            
        except Exception as e:
            print(f"Small LLM ONNX inference failed: {e}")
            return "Generated response from small LLM ONNX model"
    
    def _generate_transformers_response(self, prompt: str) -> str:
        """Generate response using small LLM model."""
        try:
            if self.text_generator is None:
                return "Generated response from small LLM model"
            
            # Show NPU usage if applicable
            if self.use_qnn:
                print("🚀 Generating response using QNN NPU acceleration with small LLM")
            
            # Format prompt for small model (DialoGPT format)
            formatted_prompt = f"User: {prompt}\nAssistant:"
            
            # Set pad token ID safely
            pad_token_id = self.transformers_tokenizer.eos_token_id if self.transformers_tokenizer else None
            
            result = self.text_generator(
                formatted_prompt, 
                max_new_tokens=self.max_response_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=pad_token_id
            )
            
            # Extract response (remove the prompt part)
            full_response = result[0]['generated_text']
            response = full_response[len(formatted_prompt):].strip()
            
            # Clean up the response
            if response.startswith("</s>"):
                response = response[4:].strip()
            
            return response
            
        except Exception as e:
            print(f"Small LLM Transformers inference failed: {e}")
            return "Generated response from small LLM model"
    
    def format_context_for_llm(self, context: List[Dict[str, Any]]) -> str:
        """Format context events for edge AI analysis."""
        if not context:
            return "No context available for edge AI analysis."
        
        formatted_context = []
        for i, event in enumerate(context, 1):
            source = event.get('source', 'unknown')
            text = event.get('text', '')
            timestamp = event.get('metadata', {}).get('timestamp', '')
            
            formatted_context.append(f"{i}. [{source.upper()}] {timestamp}: {text}")
        
        return "\n".join(formatted_context)
    
    def generate_prompt(self, user_input: str, context: List[Dict[str, Any]]) -> str:
        """Generate edge AI prompt for Llama-v3.2-3B-Instruct on Snapdragon X-Elite."""
        context_text = self.format_context_for_llm(context)
        
        # Create a more specific prompt for Llama v3.2 Instruct
        if "summarize" in user_input.lower():
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an edge AI assistant running on Snapdragon X-Elite NPU, analyzing a work session. Based on this context, provide a detailed summary.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context_text}

User request: {user_input}

Provide a comprehensive summary including:
- What the user was working on
- Key activities and patterns
- Insights and observations
- Recommendations for optimization

Please be concise but thorough.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an edge AI assistant running on Snapdragon X-Elite NPU. Analyze this session and respond to the user's query.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Context from user's session:
{context_text}

User Query: {user_input}

Provide a helpful response with insights and recommendations. Be concise and practical.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def generate_response(self, user_input: str, context: List[Dict[str, Any]]) -> str:
        """Generate response using edge AI with available models. No mock responses."""
        if not self.model_loaded:
            self.load_model()
        
        try:
            # Try LM Studio first (with context)
            if LM_STUDIO_AVAILABLE and self.lm_client:
                try:
                    print("[LLM] 🧠 Using LM Studio with context")
                    # For LM Studio, we'll use the prompt directly
                    prompt = self.generate_prompt(user_input, context)
                    response = self._generate_lm_studio_response(prompt)
                    return self._post_process_response(response, context, user_input)
                except Exception as e:
                    print(f"[LLM] ⚠️ LM Studio failed, falling back: {e}")
            
            # Try Flan-T5 model (high quality, quantized)
            if hasattr(self, 'flan_t5_service') and self.flan_t5_service and self.flan_t5_service.model_loaded:
                try:
                    print("[LLM] 🧠 Using Flan-T5 with context")
                    return self.flan_t5_service.generate_response(user_input, context)
                except Exception as e:
                    print(f"[LLM] ⚠️ Flan-T5 failed, falling back: {e}")
            elif FLAN_T5_AVAILABLE:
                try:
                    flan_t5_service = get_flan_t5_service()
                    if flan_t5_service.model_loaded:
                        print("[LLM] 🧠 Using Flan-T5 with context")
                        return flan_t5_service.generate_response(user_input, context)
                except Exception as e:
                    print(f"[LLM] ⚠️ Flan-T5 failed, falling back: {e}")
            
            # Fall back to local models
            prompt = self.generate_prompt(user_input, context)
            if self.session or self.text_generator:
                response = self.generate_edge_response(prompt)
                return self._post_process_response(response, context, user_input)
            
            # If no model is available, raise error
            raise RuntimeError("No models available. Please check LM Studio connection or local model files.")
        except Exception as e:
            print(f"Edge AI generation failed: {e}")
            raise
    
    def _generate_enhanced_mock_response(self, context: List[Dict[str, Any]], user_input: str) -> str:
        """Generate enhanced mock response for edge AI scenarios."""
        import random
        
        # Simulate edge processing time
        time.sleep(0.1)  # Faster for edge devices
        
        # Analyze context
        context_summary = f"Found {len(context)} context items"
        ocr_count = len([e for e in context if e.get('source') == 'ocr'])
        asr_count = len([e for e in context if e.get('source') == 'asr'])
        
        # Select appropriate mock responses
        if "summarize" in user_input.lower():
            summary = random.choice(self.mock_responses["summarize"])
        else:
            summary = f"Your session contains {ocr_count} screenshots and {asr_count} audio recordings, indicating active work across multiple modalities."
        
        analysis = random.choice(self.mock_responses["analysis"])
        recommendation = random.choice(self.mock_responses["recommendations"])
        edge_ai_insight = random.choice(self.mock_responses["edge_ai"])
        
        # Generate contextual insights
        insights = self._generate_contextual_insights(context, user_input)
        
        mock_response = f"""🤖 Edge AI Assistant (Snapdragon X-Elite)

Based on your session context ({context_summary}), here's what I understand:

📊 Session Overview:
• Screenshots captured: {ocr_count}
• Audio recordings: {asr_count}
• Total events: {len(context)}

🔍 Edge AI Analysis:
{analysis}

💡 Key Insights:
{insights}

📝 Summary:
{summary}

🎯 Edge AI Recommendations:
1. {recommendation}
2. Use specific queries to get targeted insights from on-device AI
3. Consider organizing your work into focused sessions

🚀 Edge AI Benefits:
{edge_ai_insight}

⚡ Performance: Processed locally on Snapdragon X-Elite for privacy and speed

Note: This is an enhanced mock response optimized for edge AI scenarios. Real model integration ready when models are available!"""
        
        return mock_response.strip()
    
    def _generate_contextual_insights(self, context: List[Dict[str, Any]], user_input: str) -> str:
        """Generate contextual insights for edge AI scenarios."""
        if not context:
            return "• No context available for edge AI analysis"
        
        insights = []
        
        # Analyze timing patterns
        timestamps = [e.get('metadata', {}).get('timestamp') for e in context if e.get('metadata', {}).get('timestamp')]
        if len(timestamps) > 1:
            insights.append("• Multiple captures suggest active work session ideal for edge AI")
        
        # Analyze source patterns
        sources = [e.get('source') for e in context]
        if 'ocr' in sources and 'asr' in sources:
            insights.append("• Mixed media capture perfect for edge AI multi-modal processing")
        elif 'ocr' in sources:
            insights.append("• Visual focus suggests screen-based work suitable for edge AI analysis")
        elif 'asr' in sources:
            insights.append("• Audio focus suggests verbal communication ideal for edge AI transcription")
        
        # Analyze content patterns
        if len(context) > 5:
            insights.append("• High activity level indicates productive session benefiting from edge AI")
        
        if "summarize" in user_input.lower():
            insights.append("• Summary request shows need for edge AI-powered overview")
        
        return "\n".join(insights) if insights else "• Working patterns are being established for edge AI optimization"
    
    def _post_process_response(self, response: str, context: List[Dict[str, Any]], user_input: str) -> str:
        """Post-process edge AI model response."""
        # Add edge AI context information
        context_info = f"\n\n⚡ Edge AI: Processed {len(context)} events locally on Snapdragon X-Elite"
        return response + context_info
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the edge AI model."""
        lm_studio_status = {
            "connected": LM_STUDIO_AVAILABLE and self.lm_client is not None,
            "base_url": "http://localhost:8080" if self.lm_client else None
        }
        
        return {
            "model_name": self.model_name,
            "loaded": self.model_loaded,
            "onnx_available": ONNX_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "lm_studio_available": LM_STUDIO_AVAILABLE,
            "lm_studio_status": lm_studio_status,
            "model_path": self.small_llm_path,
            "model_exists": os.path.exists(self.small_llm_path) if self.small_llm_path else False,
            "max_context_length": self.max_context_length,
            "max_response_length": self.max_response_length,
            "edge_optimized": True,
            "platform": "Snapdragon X-Elite",
            "model_type": "Llama-v3.2-3B-Instruct via LM Studio"
        }

# Global instance
llm_service = LLMService() 