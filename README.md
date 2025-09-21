# 🧠 Temporal LLaVA Habitat

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](../../issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

_Augmenting Vision-Language Models with Temporal Reasoning for Embodied AI_

🔗 [Demo](#-demo) · [Installation](#-installation) · [Usage](#-usage) · [Benchmarks](#-benchmarks) · [Research](#-research) · [Citation](#-citation) · [Future Work](#-future-work)

---

## ✨ Overview  

Recent advances in **vision-language models (VLMs)** like LLaVA (Large Language and Vision Assistant) have revolutionized **static image understanding**. However, embodied AI agents — whether a household robot or an aerial drone must act across **time-dependent sequences of events**.  

This repository extends [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) with **temporal reasoning capabilities**, enabling VLMs to operate in **dynamic, simulated embodied environments**.  

By augmenting LLaVA with **temporal sequence modeling**, this project demonstrates:  

- **Tracking objects across frames** instead of isolated images  
- **Inferring cause-effect relationships** in simulated worlds  
- **Predicting future states** to support planning  
- **Context-aware decision making** for embodied AI  

💡 In short: Temporal LLaVA Habitat transforms static perception into **temporal intelligence**, bridging the gap between seeing and acting in real-world robotics.  

---

## ⚡ The Challenge: Beyond Static Understanding  

Most multimodal models can answer:  
- _“What is in this picture?”_  

But they fail at:  
- _“Where did the object move?”_  
- _“What happened just before this?”_  
- _“What will happen next?”_  

This project addresses that temporal blindness. For example:  
- A service robot tracking a mug that was moved from kitchen to table.  
- A drone anticipating a moving obstacle rather than reacting too late.  

**Temporal reasoning** unlocks the next stage of embodied AI: moving from **reactive recognition** to **predictive, context-aware intelligence**.  

---

## 📂 Project Structure  

```

embodied-temporal-reasoning/
│── main.py                # Core training and evaluation scripts
│── models/                # Temporal extensions of LLaVA
│── habitat\_env/           # Embodied AI simulation environments
│── configs/               # Experiment configurations
│── benchmarks/            # Evaluation metrics and results
│── screenshots/           # Demo screenshots
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

````

---

## 🖼️ Screenshots  

| Temporal Reasoning | Habitat Simulation | Debug Logs |
|--------------------|--------------------|------------|
| ![Temporal Reasoning](screenshots/temporal_demo.png) | ![Habitat Environment](screenshots/habitat_scene.png) | ![Debug](screenshots/debug_logs.png) |

---

## ⚙️ Installation  

### 1. Clone this repository  
```bash
git clone https://github.com/Rukh-sana/embodied-temporal-reasoning.git
cd embodied-temporal-reasoning
````

### 2. Install dependencies

We build on [Habitat-Sim](https://github.com/facebookresearch/habitat-sim). Please follow the [official installation guide](https://github.com/facebookresearch/habitat-sim#installation).

After Habitat-Sim is installed, install project-specific dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

Run training/evaluation inside Habitat with temporal reasoning:

```bash
python main.py --config configs/temporal_llava.yaml
```

Available configuration files are in [`configs/`](configs).

---

## 📊 Benchmarks

Temporal LLaVA was evaluated on **Habitat embodied tasks**:

* ✅ **Object tracking across frames** → 23% improvement over static LLaVA
* ✅ **Temporal Question Answering** → Better accuracy in multi-frame queries
* ✅ **Predictive Planning** → Higher success rate in navigation tasks

Detailed benchmark tables are available in [`benchmarks/`](benchmarks).

---

## 📚 Citation

If you use this repository, please cite:

```bibtex
@article{rukhsana2024temporal,
  title={Temporal LLaVA Habitat: Augmenting Vision-Language Models with Temporal Reasoning for Embodied AI},
  author={Rukh-Sana, ...},
  journal={Preprint},
  year={2024}
}
```

---

## 🔮 Future Work

This work lays the foundation for **temporal reasoning in embodied AI**, but there are exciting future directions:

* **Aerial Robotics & Drones** – Extend temporal reasoning to aerial navigation tasks, integrating **reinforcement learning** for obstacle avoidance and trajectory prediction.
* **Transformers in Robotics** – Use transformer-based architectures for **long-horizon temporal memory**, vital for drones, service robots, and autonomous systems.
* **Cross-Modal Integration** – Combine **vision, language, proprioception, and environment maps** to improve generalization in unseen environments.
* **Real-World Deployment** – From household assistants to drone fleets, deploy temporal reasoning beyond simulation into robotics platforms.

This research aligns with my earlier contributions in **STEM and robotics research**:

* [STEM Journal Paper](https://www.lcjstem.com/index.php/jstem/article/view/101)
* [Wiley Publication](https://onlinelibrary.wiley.com/doi/abs/10.1002/2050-7038.12504)

Together, these works chart a path toward **temporal intelligence as a cornerstone of embodied AI**.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request if you’d like to extend this work.

---

## 📜 License

This project is licensed under the MIT License.

```
