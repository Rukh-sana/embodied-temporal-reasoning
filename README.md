# Temporal Multimodal Reasoning in Embodied AI

## Overview
This repository implements temporal reasoning capabilities for vision-language models in embodied AI systems, enabling coherent multi-step task execution in simulated environments.

## Research Context
This work addresses the fundamental limitation of current vision-language models that process observations frame-by-frame, lacking temporal awareness necessary for sequential task execution.

## Problem Statement
Current vision-language models in embodied AI systems lack temporal awareness, preventing coherent multi-step task execution. This research develops systematic temporal reasoning capabilities through:
- LSTM-based temporal fusion mechanisms
- Hierarchical command decomposition
- Memory-augmented state tracking
- Comprehensive evaluation frameworks

## Key Features
- **Temporal State Representation**: LSTM-based fusion for sequential observations
- **Hierarchical Command Processing**: Multi-step task decomposition and tracking
- **Memory System**: Persistent environmental state management
- **Evaluation Framework**: Comprehensive benchmarks for temporal reasoning assessment

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Habitat-Sim environment
- Ollama with LLaVA model

### Setup
```bash
git clone https://github.com/your-username/temporal-multimodal-embodied-ai.git
cd temporal-multimodal-embodied-ai
pip install -r requirements.txt

## Dataset Requirements

### ReplicaCAD Dataset
This project uses the ReplicaCAD dataset for realistic indoor environments:
- Download from: [ReplicaCAD Official Repository]
- Place in: `data/replica_cad/`
- Required files: `replicaCAD.scene_dataset_config.json`

### Scene Configuration
```python
scene_path = "data/replica_cad/replicaCAD.scene_dataset_config.json"
available_scenes = ["apt_0", "apt_1", "apt_2", "frl_apartment_0", ...]
