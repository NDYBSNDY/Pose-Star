# Pose-Star++: Unified and Training-Free Human-Centric Image Editing with Long and Complex Instructions

## 📝 Abstract

Human-centric image editing has achieved remarkable progress in single-person scenarios, where the correspondence between language instructions and visual targets is relatively straightforward. However, real-world applications such as portrait retouching, fashion design, and post-production often involve **multi-person scenes**. In these cases, editing instructions must simultaneously specify **which person to edit** and **how to edit**, leading to significantly longer and more complex language descriptions.

To address this challenge, we propose **Pose-Star++**, a unified, training-free framework for human-centric image editing driven by long and complex instructions. Pose-Star++ integrates a large vision-language model for high-level semantic understanding with a bidirectional alignment mechanism that enforces structural consistency at the pose level. This design enables precise target selection and spatially accurate edits across diverse editing scenarios, while remaining fully plug-and-play with existing diffusion-based image editing models.

Extensive experiments demonstrate that Pose-Star++ significantly improves robustness to instruction length and complexity, while maintaining competitive visual quality and strong generalization across multiple human-centric editing tasks.

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
