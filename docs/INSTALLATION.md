# Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ / macOS 11+ / Windows 10+
- **Python**: 3.8 - 3.11
- **RAM**: 16GB minimum
- **Storage**: 50GB free space
- **GPU**: Optional but recommended (NVIDIA with 8GB+ VRAM)

### Recommended Setup
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.9
- **RAM**: 32GB
- **GPU**: NVIDIA RTX 3080/4080 or A100
- **CUDA**: 11.8+

## Prerequisites

Before starting, ensure you have:
- Git installed
- Conda or Miniconda ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- NVIDIA drivers (if using GPU)

## Step-by-Step Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/temporal-llava-habitat.git
cd temporal-llava-habitat

## System Requirements
[Your requirements content here]

<details>
<summary><b>2. Create Environment</b></summary>
```bash
# Create conda environment
conda create -n temporal-llava python=3.9
conda activate temporal-llava

<details>
<summary><b>3. Install project dependencies</b></summary>
```bash
pip install -r requirements.txt
Install in development mode
pip install -e .
