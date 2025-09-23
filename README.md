cat > README.md << 'EOF'
# Temporal LLaVA Habitat Integration

Temporal reasoning implementation for embodied AI using LLaVA and Habitat-Sim.

## Performance Results

| Metric | Achieved | Target | Status |
|--------|----------|--------|--------|
| Task Success Rate | 100.0% | 85% | ✅ Exceeded |
| Action Latency | 59.9ms | <100ms | ✅ Achieved |
| Memory Integration | Active | Active | ✅ Functional |

## Files

- `habitat_llava_temporal.py` - Enhanced temporal reasoning implementation (44KB)
- `habitat_llava_integration.py` - Original LLaVA-Habitat integration (21KB)
- `run_demo.py` - Demo execution script
- `temporal_results_1758646054.json` - Complete performance results

## Quick Start
```bash
# Install dependencies
pip install habitat-sim requests pillow numpy opencv-python

# Start Ollama
ollama serve
ollama pull llava:latest

# Run temporal reasoning demo
python run_demo.py
