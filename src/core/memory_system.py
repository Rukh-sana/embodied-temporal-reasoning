"""
Memory System for Persistent State Tracking
"""

from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SubGoal:
    """Representation of a sub-goal in multi-step tasks"""
    description: str
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class EnvironmentalMemory:
    """Memory of environmental states and objects"""
    objects_seen: Dict[str, Dict] = None
    spatial_map: Dict[str, Any] = None
    last_positions: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.objects_seen is None:
            self.objects_seen = {}
        if self.spatial_map is None:
            self.spatial_map = {}
        if self.last_positions is None:
            self.last_positions = []

class HierarchicalMemorySystem:
    """Hierarchical memory system for temporal reasoning"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        
        # Short-term memory (recent observations and actions)
        self.short_term_buffer = []
        
        # Task memory (current and recent tasks)
        self.current_task = None
        self.task_history = []
        self.sub_goals = []
        
        # Environmental memory (persistent world state)
        self.environmental_memory = EnvironmentalMemory()
        
        # Episodic memory (important events and sequences)
        self.episodic_memory = []
        
    def update_short_term(self, observation: Dict[str, Any]):
        """Update short-term memory with new observation"""
        self.short_term_buffer.append(observation)
        
        # Maintain buffer size
        if len(self.short_term_buffer) > self.max_history:
            self.short_term_buffer.pop(0)
    
    def set_current_task(self, task_description: str, sub_goals: List[str]):
        """Set current task and decompose into sub-goals"""
        self.current_task = task_description
        self.sub_goals = [SubGoal(description=goal) for goal in sub_goals]
    
    def update_sub_goal_status(self, goal_index: int, status: TaskStatus):
        """Update status of a specific sub-goal"""
        if 0 <= goal_index < len(self.sub_goals):
            self.sub_goals[goal_index].status = status
    
    def get_current_sub_goal(self) -> Optional[SubGoal]:
        """Get the next pending sub-goal"""
        for goal in self.sub_goals:
            if goal.status == TaskStatus.PENDING:
                return goal
        return None
    
    def update_environmental_memory(self, objects: Dict[str, Any], position: np.ndarray):
        """Update environmental memory with new information"""
        # Update object memory
        for obj_name, obj_info in objects.items():
            self.environmental_memory.objects_seen[obj_name] = obj_info
        
        # Update position history
        self.environmental_memory.last_positions.append(position)
        if len(self.environmental_memory.last_positions) > 20:
            self.environmental_memory.last_positions.pop(0)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context for decision making"""
        return {
            "current_task": self.current_task,
            "active_sub_goal": self.get_current_sub_goal(),
            "completed_sub_goals": [g for g in self.sub_goals if g.status == TaskStatus.COMPLETED],
            "recent_observations": self.short_term_buffer[-5:] if self.short_term_buffer else [],
            "known_objects": list(self.environmental_memory.objects_seen.keys()),
            "memory_size": len(self.short_term_buffer)
        }
    
    def clear_task_memory(self):
        """Clear current task and sub-goals"""
        if self.current_task:
            self.task_history.append({
                "task": self.current_task,
                "sub_goals": self.sub_goals,
                "completion_status": [g.status for g in self.sub_goals]
            })
        
        self.current_task = None
        self.sub_goals = []