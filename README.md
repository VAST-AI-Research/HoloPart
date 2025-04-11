# HoloPart: Generative 3D Part Amodal Segmentation

<div align="center">

[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-blue.svg)](https://vast-ai-research.github.io/HoloPart)
[![Paper](https://img.shields.io/badge/📑-Paper-green.svg)](https://arxiv.org/abs/2504.07943)
[![Model](https://img.shields.io/badge/🤗-Model-yellow.svg)](https://huggingface.co/VAST-AI/HoloPart)
[![Online Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/VAST-AI/HoloPart)

</div>

![teaser](assets/doc/teaser.png)

Generative 3D part amodal segmentation--decomposing a 3D shape into complete, semantically meaningful parts.

## 🔥 Updates

### 📅 April 2025
- 🚀 **Initial Release**: Published code, pretrained models, and interactive demo.
- 📌 **Coming Soon**: 
  - [ ] Integration of segmentation methods into the HoloPart pipeline.

## 🔨 Installation

Clone the repo:
```bash
git clone https://github.com/VAST-AI-Research/HoloPart.git
cd HoloPart
```

Create a conda environment (optional):
```bash
conda create -n holopart python=3.10
conda activate holopart
```

Install dependencies:
```bash
# pytorch (select correct CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/{your-cuda-version}

# other dependencies
pip install -r requirements.txt
```

## 💡 Quick Start

### Step 1: Prepare segmented mesh
Upload a mesh with part segmentation. We recommend using these segmentation tools:
- [SAMPart3D](https://github.com/Pointcept/SAMPart3D)
- [SAMesh](https://github.com/gtangg12/samesh)

For a mesh file `mesh.glb` and corresponding face mask `mask.npy`, prepare your input using this Python code:
```python
import trimesh
import numpy as np
mesh = trimesh.load("mesh.glb", force="mesh")
mask_npy = np.load("mask.npy")
mesh_parts = []
for part_id in np.unique(mask_npy):
    mesh_part = mesh.submesh([mask_npy == part_id], append=True)
    mesh_parts.append(mesh_part)
mesh_parts = trimesh.Scene(mesh_parts).export("input_mesh.glb")
```
The resulting **input_mesh.glb** is the prepared input for HoloPart.

### Step 2: Decompose the 3D mesh into complete parts:
```bash
python -m scripts.inference_holopart --mesh-input assets/example_data/000.glb
```

The required model weights will be automatically downloaded:
- HoloPart model from [VAST-AI/HoloPart](https://huggingface.co/VAST-AI/HoloPart) → `pretrained_weights/HoloPart`

## ⭐ Acknowledgements

We would like to thank the following open-source projects and research works that made HoloPart possible:

- [🤗 Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [HunyuanDiT](https://github.com/Tencent/HunyuanDiT) for DiT
- [FlashVDM](https://github.com/Tencent/FlashVDM) for their lightning vecset decoder
- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) for 3D shape representation
- [TripoSG](https://github.com/VAST-AI-Research/TripoSG) as our base model

We are grateful to the broader research community for their open exploration and contributions to the field of 3D generation.

## 📚 Citation

```
@article{yang2025holopart,
      title={HoloPart: Generative 3D Part Amodal Segmentation}, 
      author={Yang, Yunhan and Guo, Yuan-Chen and Huang, Yukun and Zou, Zi-Xin and Yu, Zhipeng and Li, Yangguang and Cao, Yan-Pei and Liu, Xihui},
      journal={arXiv preprint arXiv:2504.07943},
      year={2025}
}
```
