"""
Temporal Fusion Module - Core component for temporal reasoning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple

class TemporalFusionLSTM(nn.Module):
    """LSTM-based temporal fusion for sequential observations"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_layers: int = 1):
        super(TemporalFusionLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal fusion
        
        Args:
            sequence_embeddings: Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Fused temporal representation: (batch_size, hidden_dim)
        """
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(sequence_embeddings)
        
        # Attention over LSTM outputs
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Use the last attended output
        temporal_context = attended_out[:, -1, :]
        
        # Final projection
        output = self.output_proj(temporal_context)
        
        return output

class TemporalBuffer:
    """Buffer for storing and managing temporal observations"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer = []
        self.embeddings = []
        self.metadata = []
        
    def add_observation(self, observation: np.ndarray, embedding: Optional[np.ndarray] = None, metadata: Optional[Dict] = None):
        """Add new observation to buffer"""
        self.buffer.append(observation)
        
        if embedding is not None:
            self.embeddings.append(embedding)
            
        if metadata is not None:
            self.metadata.append(metadata)
        
        # Maintain buffer size
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            if self.embeddings:
                self.embeddings.pop(0)
            if self.metadata:
                self.metadata.pop(0)
    
    def get_sequence(self, length: Optional[int] = None) -> List[np.ndarray]:
        """Get observation sequence"""
        if length is None:
            return self.buffer.copy()
        else:
            return self.buffer[-length:] if len(self.buffer) >= length else self.buffer.copy()
    
    def get_embeddings(self, length: Optional[int] = None) -> List[np.ndarray]:
        """Get embedding sequence"""
        if not self.embeddings:
            return []
        
        if length is None:
            return self.embeddings.copy()
        else:
            return self.embeddings[-length:] if len(self.embeddings) >= length else self.embeddings.copy()
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.embeddings.clear()
        self.metadata.clear()
    
    def __len__(self):
        return len(self.buffer)