#!/usr/bin/env python3
"""
Enhanced Habitat Simulator with Temporal Reasoning Capabilities
"""

import argparse
import os
import base64
import requests
import json
import time
import io
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
import cv2
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

class TemporalHabitatSimulator:
    """
    Enhanced Habitat Simulator with temporal reasoning capabilities
    """
    
    def __init__(self, scene_path: str):
        """Initialize the temporal habitat simulator"""
        self.scene_path = scene_path
        self.simulator = None
        self.agent_id = 0
        
        # Temporal components (to be implemented)
        self.frame_buffer = []
        self.max_history = 10
        self.temporal_state = None
        
        self.setup_environment()
        self.setup_simulator()
        
    def setup_environment(self):
        """Set up environment variables"""
        os.environ["MAGNUM_GPU_VALIDATION"] = "OFF"
        os.environ["HABITAT_SIM_LOG"] = "quiet"
        
    def setup_simulator(self):
        """Initialize Habitat-Sim simulator"""
        sim_settings = {
            "width": 640,
            "height": 480,
            "scene_dataset": self.scene_path,
            "scene": "apt_0",
            "default_agent": 0,
            "sensor_height": 1.5,
            "color_sensor": True,
            "seed": 1,
            "enable_physics": False,
        }

        cfg = self.make_simple_cfg(sim_settings)
        
        try:
            self.simulator = habitat_sim.Simulator(cfg)
            self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])
            
            # Set initial agent state
            agent_state = habitat_sim.AgentState()
            agent_state.position = np.array([0.0, 0.0, 0.0])
            self.agent.set_state(agent_state)
            
        except Exception as e:
            print(f"Failed to initialize simulator: {e}")
            raise
    
    def make_simple_cfg(self, settings):
        """Create simulator configuration"""
        # Simulator configuration
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
        sim_cfg.scene_id = settings["scene"]
        sim_cfg.enable_physics = settings["enable_physics"]
        sim_cfg.gpu_device_id = -1
        sim_cfg.force_separate_semantic_scene_graph = True

        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # Create RGB sensor
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        
        agent_cfg.sensor_specifications = [color_sensor_spec]

        # Define action space
        action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        agent_cfg.action_space = action_space

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    def get_observation(self) -> Optional[np.ndarray]:
        """Get current observation with temporal buffering"""
        obs = self.simulator.get_sensor_observations()
        rgb_img = obs["color_sensor"]
        
        # Handle RGBA to RGB conversion
        if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 4:
            rgb_img = rgb_img[:, :, :3]
        
        # Convert to uint8
        if rgb_img.dtype != np.uint8:
            if rgb_img.max() <= 1.0:
                rgb_img = (rgb_img * 255).astype(np.uint8)
            else:
                rgb_img = rgb_img.astype(np.uint8)
        
        # Add to temporal buffer
        self.add_to_buffer(rgb_img)
        
        return rgb_img
    
    def add_to_buffer(self, observation: np.ndarray):
        """Add observation to temporal buffer"""
        self.frame_buffer.append(observation)
        if len(self.frame_buffer) > self.max_history:
            self.frame_buffer.pop(0)
    
    def perform_action(self, action_name: str) -> bool:
        """Perform action in the environment"""
        try:
            if action_name in ["move_forward", "turn_left", "turn_right"]:
                self.simulator.step(action_name)
                return True
            return False
        except Exception as e:
            print(f"Error performing action {action_name}: {e}")
            return False
    
    def get_temporal_context(self) -> Dict:
        """Get temporal context (placeholder for future implementation)"""
        return {
            "buffer_size": len(self.frame_buffer),
            "max_history": self.max_history,
            "current_step": len(self.frame_buffer)
        }
    
    def close(self):
        """Close the simulator"""
        if self.simulator:
            self.simulator.close()