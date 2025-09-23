#!/usr/bin/env python3

import argparse
import os
import base64
import requests
import json
import time
import io

import magnum as mn
import numpy as np
import cv2
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

# Use the scene path you provided
scene_path = "/home/dell/Research Proposal/data/replica_cad/replicaCAD.scene_dataset_config.json"

def make_simple_cfg(settings):
    # This is the modern way to configure the simulator
    
    # Simulator configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.gpu_device_id = -1  # CRITICAL: Use CPU rendering by setting to -1
    sim_cfg.force_separate_semantic_scene_graph = True

    # Agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # Create the RGB sensor
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    
    agent_cfg.sensor_specifications = [color_sensor_spec]

    # Define the action space
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

class LLaVAHabitatSimulator:
    def __init__(self):
        # Basic setup
        self.simulator = None
        self.agent_id = 0
        self.ollama_url = "http://localhost:11434"
        self.model = "llava:latest"
        
        # Command system attributes
        self.last_response = ""
        self.command_mode = False
        self.current_command = ""
        
        print("🔧 Initializing LLaVA Habitat Simulator...")
        self.setup_environment()
        self.setup_simulator()
        self.check_ollama_connection()
        print("✅ Simulator initialized successfully!")
        
    def setup_environment(self):
        """Set up environment variables"""
        print("🖥️ Setting up environment...")
        os.environ["MAGNUM_GPU_VALIDATION"] = "OFF"
        os.environ["HABITAT_SIM_LOG"] = "quiet"
        
    def setup_simulator(self):
        """Initialize Habitat-Sim simulator using the basic_sim configuration"""
        print("🏠 Setting up Habitat simulator...")
        
        # 1. Create simulation settings (from basic_sim.py)
        sim_settings = {
            "width": 640,
            "height": 480,
            "scene_dataset": scene_path,
            "scene": "apt_0",  # Try "apt_1", "apt_2", etc. if this fails
            "default_agent": 0,
            "sensor_height": 1.5,
            "color_sensor": True,
            "seed": 1,
            "enable_physics": False,
        }

        # 2. Create the configuration using the modern method
        cfg = make_simple_cfg(sim_settings)
        
        # 3. Create the simulator
        try:
            self.simulator = habitat_sim.Simulator(cfg)
            print("Simulator created successfully!")
        except Exception as e:
            print(f"Failed to create simulator: {e}")
            print("Please check your scene path and scene name.")
            # Try a different scene if apt_0 doesn't work
            sim_settings["scene"] = "apt_1"
            print(f"Trying scene: {sim_settings['scene']}")
            cfg = make_simple_cfg(sim_settings)
            try:
                self.simulator = habitat_sim.Simulator(cfg)
                print("Simulator created successfully with apt_1!")
            except Exception as e2:
                print(f"Also failed with apt_1: {e2}")
                raise

        # 4. Get a reference to the agent
        self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])
        
        # 5. Set the agent's initial state - use a safer starting position
        agent_state = habitat_sim.AgentState()
        # Try a position that's more likely to be valid
        agent_state.position = np.array([0.0, 0.0, 0.0])
        self.agent.set_state(agent_state)

    def check_ollama_connection(self):
        """Check Ollama connection with detailed debugging"""
        print(f"🔍 Checking Ollama connection at: {self.ollama_url}")
        
        try:
            # Test basic connection
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                model_names = [model['name'] for model in models]
                print(f"Available models: {model_names}")
                
                if self.model in model_names:
                    print(f"✓ Connected to Ollama - Using model: {self.model}")
                    return True
                else:
                    print(f"❌ Model {self.model} not found. Available: {model_names}")
                    return False
            else:
                print(f"❌ Ollama HTTP error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error: {e}")
            print("💡 Try: ollama serve")
            return False
        except requests.exceptions.Timeout as e:
            print(f"❌ Timeout error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    def get_observation(self):
        """Get current observation and convert to proper format"""
        obs = self.simulator.get_sensor_observations()
        rgb_img = obs["color_sensor"]
        
        # Handle RGBA to RGB conversion
        if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 4:
            rgb_img = rgb_img[:, :, :3]  # Remove alpha channel
        
        # Convert to uint8 if needed
        if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float64:
            if rgb_img.max() <= 1.0:
                rgb_img = (rgb_img * 255).astype(np.uint8)
            else:
                rgb_img = rgb_img.astype(np.uint8)
        elif rgb_img.dtype != np.uint8:
            rgb_img = rgb_img.astype(np.uint8)
        
        return rgb_img

    def perform_action(self, action_name):
        """Perform an action using the simulator's action system"""
        try:
            if action_name == "move_forward":
                obs = self.simulator.step("move_forward")
            elif action_name == "turn_left":
                obs = self.simulator.step("turn_left")
            elif action_name == "turn_right":
                obs = self.simulator.step("turn_right")
            else:
                return False
                
            return True
            
        except Exception as e:
            print(f"Error performing action {action_name}: {e}")
            return False

    def image_to_base64(self, image_array):
        """Convert image to base64"""
        try:
            pil_image = Image.fromarray(image_array)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None

    def query_llava(self, image_array, prompt):
        """Query LLaVA with image and prompt - with robust error handling"""
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
            
            # Use longer timeout and add retry logic
            for attempt in range(3):
                try:
                    response = requests.post(
                        f"{self.ollama_url}/api/generate", 
                        json=payload, 
                        timeout=60,  # Increased timeout
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result.get("response", "No response received")
                    else:
                        print(f"HTTP Error {response.status_code}: {response.text}")
                        if attempt < 2:  # Retry
                            print(f"Retrying... (attempt {attempt + 2}/3)")
                            time.sleep(2)
                            continue
                        return f"HTTP Error: {response.status_code}"
                        
                except requests.exceptions.Timeout:
                    print(f"Timeout on attempt {attempt + 1}/3")
                    if attempt < 2:
                        time.sleep(3)
                        continue
                    return "Error: Request timed out after 3 attempts"
                    
                except requests.exceptions.ConnectionError as e:
                    print(f"Connection error on attempt {attempt + 1}/3: {e}")
                    if attempt < 2:
                        time.sleep(3)
                        continue
                    return "Error: Could not connect to Ollama. Make sure 'ollama serve' is running."
                    
        except Exception as e:
            return f"Error: {str(e)}"

    def execute_command(self, command, current_img):
        """Execute natural language commands using LLaVA"""
        try:
            prompt = f"""
            You are an AI agent in a 3D environment. The user gave you this command: "{command}"
            
            First, describe what you see in the current image.
            Then, based on the command and what you see, suggest what action I should take.
            
            Available actions:
            - move_forward: Move forward
            - turn_left: Turn left  
            - turn_right: Turn right
            - look_around: Describe what you see
            
            Format your response as:
            OBSERVATION: [what you see]
            ACTION: [suggested action]
            REASONING: [why this action]
            """
            
            response = self.query_llava(current_img, prompt)
            
            # Parse the response to extract suggested action
            suggested_action = self.parse_action_from_response(response)
            
            # Execute the suggested action
            if suggested_action and suggested_action in ["move_forward", "turn_left", "turn_right"]:
                print(f"🤖 Executing action: {suggested_action}")
                self.perform_action(suggested_action)
                return f"✓ Executed: {suggested_action}\n\n{response}"
            else:
                return response
                
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def parse_action_from_response(self, response):
        """Parse suggested action from LLaVA response"""
        try:
            lines = response.lower().split('\n')
            for line in lines:
                if 'action:' in line:
                    if 'move_forward' in line or 'move forward' in line:
                        return 'move_forward'
                    elif 'turn_left' in line or 'turn left' in line:
                        return 'turn_left'
                    elif 'turn_right' in line or 'turn right' in line:
                        return 'turn_right'
            return None
        except:
            return None
    
    def wrap_text(self, text, max_width=50):
        """Wrap text for display"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def draw_text_overlay(self, img, text, position=(10, 100), max_lines=15):
        """Draw text overlay on image with background"""
        if not text:
            return img
            
        lines = self.wrap_text(text, max_width=50)
        lines = lines[:max_lines]  # Limit number of lines
        
        # Calculate text area dimensions
        line_height = 20
        text_height = len(lines) * line_height + 20
        text_width = max(400, max(len(line) * 7 for line in lines) + 20)
        
        # Create semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, position, 
                     (position[0] + text_width, position[1] + text_height), 
                     (0, 0, 0), -1)  # Black background
        
        # Blend overlay
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Draw text
        y_offset = position[1] + 20
        for line in lines:
            cv2.putText(img, line, (position[0] + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += line_height
            
        return img

    def run_interactive(self):
        """Run interactive demo with command support"""
        print("\n🤖 INTERACTIVE LLAVA COMMAND AGENT")
        print("=" * 60)
        print("CONTROLS:")
        print("  w/a/d - Manual movement (forward/left/right)")
        print("  SPACE - Quick scene description")
        print("  c - Enter command mode (natural language)")
        print("  r - Reset/clear response")
        print("  q - Quit")
        print("=" * 60)
        print("COMMAND EXAMPLES:")
        print("  'explore the room'")
        print("  'find the painting'") 
        print("  'look for furniture'")
        print("  'turn around and describe what you see'")
        print("=" * 60)
        
        # Test initial observation
        print("\n📸 Testing initial observation...")
        current_img = self.get_observation()
        if current_img is None:
            print("❌ Failed to get observation")
            return
        
        print("✓ Observation successful!")
        
        # Create OpenCV window
        window_name = 'LLaVA Command Agent'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 900, 700)
        
        step_count = 0
        
        print("\n🎮 Interactive mode started!")
        print("💡 Try pressing 'c' and typing a command!")
        
        try:
            while True:
                # Get current view
                current_img = self.get_observation()
                if current_img is None:
                    print("❌ Failed to get observation")
                    break
                
                # Convert for display (RGB -> BGR for OpenCV)
                display_img = cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR)
                
                # Add status overlay
                cv2.putText(display_img, f"Step: {step_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if self.command_mode:
                    cv2.putText(display_img, f"COMMAND: {self.current_command}_", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(display_img, "Press 'c' for commands, SPACE for description", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw LLaVA response overlay
                if self.last_response:
                    display_img = self.draw_text_overlay(display_img, self.last_response)
                
                # Show image
                cv2.imshow(window_name, display_img)
                
                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF
                
                if self.command_mode:
                    # Handle command input
                    if key == 13 or key == 10:  # Enter
                        if self.current_command.strip():
                            print(f"\n🎯 Executing command: '{self.current_command}'")
                            self.last_response = "Processing command..."
                            
                            # Execute command
                            response = self.execute_command(self.current_command, current_img)
                            self.last_response = f"COMMAND: {self.current_command}\n\n{response}"
                            step_count += 1
                            
                        self.command_mode = False
                        self.current_command = ""
                        
                    elif key == 27:  # ESC - exit command mode
                        self.command_mode = False
                        self.current_command = ""
                        
                    elif key == 8 or key == 127:  # Backspace
                        self.current_command = self.current_command[:-1]
                        
                    elif 32 <= key <= 126:  # Printable characters
                        self.current_command += chr(key)
                
                else:
                    # Handle normal controls
                    if key == ord('q'):
                        print("\n👋 Quitting...")
                        break
                        
                    elif key == ord('c'):
                        print("\n💬 Command mode activated! Type your command and press Enter:")
                        self.command_mode = True
                        self.current_command = ""
                        
                    elif key == ord('r'):
                        print("🔄 Clearing response...")
                        self.last_response = ""
                        
                    elif key == ord('w'):
                        print("🚶 Moving forward...")
                        if self.perform_action("move_forward"):
                            step_count += 1
                            self.last_response = "✓ Moved forward"
                        
                    elif key == ord('a'):
                        print("↺ Turning left...")
                        if self.perform_action("turn_left"):
                            step_count += 1
                            self.last_response = "✓ Turned left"
                        
                    elif key == ord('d'):
                        print("↻ Turning right...")
                        if self.perform_action("turn_right"):
                            step_count += 1
                            self.last_response = "✓ Turned right"
                        
                    elif key == ord(' '):  # SPACE
                        print("\n🧠 Getting scene description...")
                        self.last_response = "Analyzing scene..."
                        
                        prompt = "Describe what you see in this image in detail. Include objects, colors, spatial layout, and any notable features."
                        response = self.query_llava(current_img, prompt)
                        self.last_response = f"SCENE DESCRIPTION:\n\n{response}"
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error in interactive loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            if self.simulator:
                self.simulator.close()
            print("✅ Demo finished!")

def main():
    """Main function to run the LLaVA Habitat Simulator"""
    print("=" * 60)
    print("🌍 LLaVA Habitat Simulator")
    print("=" * 60)
    
    # Create simulator instance
    simulator = LLaVAHabitatSimulator()
    
    # Run interactive demo
    simulator.run_interactive()

if __name__ == "__main__":
    main()