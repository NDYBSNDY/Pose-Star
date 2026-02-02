# Pose-Star: Anatomy-Aware Editing for Open-World Fashion Images
## 📝 Abstract

To advance real-world fashion image editing, we analyze existing two-stage pipelines—mask generation followed by diffusion-based editing—which overly prioritize generator optimization while neglecting mask controllability. This results in two critical limitations: I) poor user-defined flexibility (coarse-grained human masks restrict edits to predefined regions like upper torso; fine-grained clothes masks preserve poses but forbid style/length customization). II) weak pose robustness (mask generators fail due to articulated poses and miss rare regions like waist, while human parsers remain limited by predefined categories). To address these gaps, we propose Pose-Star, a framework that dynamically recomposes body structures (e.g., neck, chest, etc.) into anatomy-aware masks (e.g., chest-length) for userdefined edits. In Pose-Star, we calibrate diffusion-derived attention (Star tokens) via skeletal keypoints to enhance rare structure localization in complex poses, suppress noise through phase-aware analysis of attention dynamics (Convergence→Stabilization→Divergence) with threshold masking and sliding-window fusion, and refine edges via cross-self attention merging and Canny alignment. This work bridges controlled benchmarks and open-world demands, pioneering anatomy-aware, pose-robust editing and laying the foundation for industrial fashion image editing.

---

## ⚙️ Environment Setup

### Requirements

- **Python** ≥ 3.9  
- **PyTorch** ≥ 1.13  
- **CUDA** ≥ 11.6 (recommended for GPU acceleration)

### Installation

We recommend creating a virtual environment before installation.

```bash
conda create -n pose_star python=3.9 -y
conda activate pose_star
```

Install PyTorch (adjust CUDA version if needed):

```bash
pip install torch torchvision torchaudio
```
