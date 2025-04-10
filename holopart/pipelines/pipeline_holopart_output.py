from dataclasses import dataclass
import trimesh
from typing import List, Optional, Union

import torch
from diffusers.utils import BaseOutput


@dataclass
class HoloPartPipelineOutput(BaseOutput):
    r"""
    Output class for HoloPart pipelines.
    """

    samples: torch.Tensor
    meshes: List[trimesh.Trimesh]