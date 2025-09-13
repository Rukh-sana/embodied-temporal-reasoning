"""
LLaVA Integration with Temporal Reasoning Support
"""

import requests
import json
import base64
import io
from typing import Dict, List, Optional
from PIL import Image
import numpy as np

class LLaVAInterface:
    """Interface for LLaVA model with temporal reasoning capabilities"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llava:latest"):
        self.ollama_url = ollama_url
        self.model = model
        self.temporal_context = []
        
    def check_connection(self) -> bool:
        """Check connection to Ollama service"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return self.model in models
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def image_to_base64(self, image_array: np.ndarray) -> Optional[str]:
        """Convert image array to base64 string"""
        try:
            pil_image = Image.fromarray(image_array)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    
    def query_single_frame(self, image_array: np.ndarray, prompt: str) -> str:
        """Query LLaVA with single frame (baseline approach)"""
        try:
            image_b64 = self.image_to_base64(image_array)
            if not image_b64:
                return "Error: Could not convert image"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate", 
                json=payload, 
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response received")
            else:
                return f"HTTP Error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def query_temporal_sequence(self, image_sequence: List[np.ndarray], prompt: str) -> str:
        """
        Query LLaVA with temporal sequence (future implementation)
        Currently uses single frame - to be enhanced with temporal reasoning
        """
        # TODO: Implement temporal fusion
        # For now, use the latest frame
        if image_sequence:
            return self.query_single_frame(image_sequence[-1], prompt)
        return "Error: No images in sequence"
    
    def update_temporal_context(self, observation: Dict):
        """Update temporal context for reasoning"""
        self.temporal_context.append(observation)
        # Keep only recent history
        if len(self.temporal_context) > 10:
            self.temporal_context.pop(0)