#!/usr/bin/env python3
"""
Run baseline LLaVA-Habitat integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from habitat_integration.simulator import TemporalHabitatSimulator
from llava_integration.vision_language_model import LLaVAInterface

def main():
    """Main function to run baseline system"""
    # Default scene path - update this to your actual path
    scene_path = "/path/to/your/replica_cad/replicaCAD.scene_dataset_config.json"
    
    print("Initializing Baseline LLaVA-Habitat System...")
    
    try:
        # Initialize components
        simulator = TemporalHabitatSimulator(scene_path)
        llava = LLaVAInterface()
        
        # Check connections
        if not llava.check_connection():
            print("Error: Could not connect to LLaVA. Make sure Ollama is running.")
            return
        
        print("System initialized successfully!")
        print("This is the baseline system - temporal reasoning not yet implemented.")
        
        # Basic demonstration
        observation = simulator.get_observation()
        if observation is not None:
            response = llava.query_single_frame(observation, "Describe what you see in this image.")
            print(f"LLaVA Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        return
    finally:
        simulator.close()

if __name__ == "__main__":
    main()