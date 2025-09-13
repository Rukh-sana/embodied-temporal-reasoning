# Temporal LLaVA Habitat


<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*Augmenting Vision-Language Models with Temporal Reasoning for Embodied AI*

[Demo](#demo) • [Installation](#installation) • [Usage](#usage) • [Research](#research-context) • [Citation](#citation)

</div>

## Overview

This repository implements temporal reasoning capabilities for vision-language models in simulated embodied environments, addressing the critical limitation of frame-by-frame processing in current multimodal AI systems.The field of embodied artificial intelligence stands at a critical juncture. While current vision-language models like LLaVA (Large Language and Vision Assistant) have achieved remarkable success in static image understanding, they face significant limitations when deployed in dynamic, real-world environments where temporal reasoning is essential. This research addresses a fundamental gap: how can we augment vision-language models with temporal reasoning capabilities to enable more sophisticated embodied AI systems?

### The Challenge: Beyond Static Understanding
Current vision-language models like LLaVA excel at static image interpretation but fail in embodied AI scenarios requiring sequential task execution. This research develops systematic temporal reasoning integration.Current multimodal AI systems excel at analyzing individual images and answering questions about static scenes. However, embodied agents operating in real environments must understand temporal sequences, track objects across time, predict future states, and make decisions based on historical context. This temporal blindness represents a critical bottleneck for practical embodied AI applications.
Consider a household robot tasked with "finding the keys that were left on the kitchen counter earlier." Current vision-language models can identify keys and counters in individual frames but cannot maintain temporal context about where objects were previously located or how scenes have changed over time. This limitation severely constrains their utility in real-world scenarios.

### Research Objectives
This research proposal aims to develop Temporal LLaVA Habitat, a novel framework that integrates temporal reasoning capabilities into vision-language models for embodied AI applications. The key objectives include:
### 1. Temporal Memory Integration
Developing architectures that can maintain and reason over temporal sequences of visual observations, enabling models to understand how scenes evolve over time.
### 2. Multi-Scale Temporal Reasoning
Implementing mechanisms to handle different temporal scales - from immediate frame-to-frame changes to long-term behavioral patterns and environmental dynamics.
### 3.Embodied Task Performance
Validating the approach on realistic embodied AI tasks in simulated environments, demonstrating improved performance on temporally-dependent challenges.

## Simulation Environment
Leveraging AI Habitat simulation platform provides:

- Controlled experimental conditions for systematic evaluation
- Diverse scenarios for robust temporal reasoning assessment
- Standardized benchmarks for comparing against existing approaches
- Scalable training environments for large-scale model development

## Evaluation Framework
Comprehensive evaluation across multiple dimensions:

### Temporal Understanding: 
Assessing ability to track changes over time
### Memory Efficiency: 
Evaluating computational and storage requirements
### Task Performance: 
Measuring improvement on embodied AI benchmarks
### Generalization: 
Testing transfer to unseen environments and tasks

## Broader Impact and Applications
The implications of this research extend across multiple domains:
### Robotics: 
Enabling robots to better understand and navigate dynamic environments by maintaining temporal context of their observations.
### Autonomous Systems: 
Improving decision-making in autonomous vehicles and drones through better temporal scene understanding.
###Human-AI Interaction: 
Facilitating more natural interactions by enabling AI systems to remember and reason about past interactions and environmental changes.
### Scientific Discovery: 
Supporting research in fields requiring temporal analysis of visual data, from biology to astronomy.

## Preliminary Results and Validation

### System Implementation
Our initial prototype successfully integrates LLaVA with AI Habitat simulation environment, enabling embodied agents to perform vision-language tasks with temporal context.

![System Interface](images/1.png)
*Figure 1: ReplicaCAD LLaVA Agent interface showing step-by-step interaction*

### Scene Understanding Capabilities
The system demonstrates sophisticated scene analysis, providing detailed descriptions that go beyond static object detection to include spatial relationships and contextual reasoning.

![Scene Description](images/2.png)
*Figure 2: Agent providing detailed scene description with spatial reasoning (Step 38)*

### Temporal Navigation Tasks
Most significantly, the agent successfully maintains context across multiple steps, demonstrating temporal reasoning in navigation tasks.

![Navigation Sequence](images/3.png)
*Figure 3: Sequential frames showing agent maintaining task context while navigating*
![Navigation Sequence](images/4.png)
*Figure 3: Sequential frames showing agent maintaining task context while navigating*
![Description](images/screenshot1.png)

### Key Findings
- Agent successfully maintains temporal context across 90+ interaction steps
- Demonstrates understanding of spatial relationships and environmental changes
- Shows reasoning capabilities that connect observations to task objectives
- Successfully executes multi-step navigation tasks requiring memory of previous states


## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB RAM minimum
- Conda/Miniconda

### Installation

<details>
<summary>📋 Complete Installation Guide</summary>

#### 1. Clone Repository
```bash
git clone https://github.com/your-username/temporal-llava-habitat.git
cd temporal-llava-habitat

