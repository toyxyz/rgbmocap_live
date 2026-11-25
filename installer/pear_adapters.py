"""Windows live adapter for pinned PEAR sources; never alter learned forwards."""
from pathlib import Path
from textwrap import indent

def adapt(folder):
    folder=Path(folder)
    def change(name,old,new):
        p=folder/name;s=p.read_text(encoding='utf8')
        if old not in s:raise ValueError('Unsupported PEAR source: '+name)
        notice='# Modified by RGB MoCap: Windows live inference; original PEAR license retained.\n'
        updated=s.replace(old,new)
        if not updated.startswith(notice):updated=notice+updated
        p.write_text(updated,encoding='utf8')
    def guard_uv(name, first, last=None):
        source=(folder/name).read_text(encoding='utf8')
        start=source.index('        '+first)
        end=source.index('\n', source.index('        '+(last or first),start))
        block=source[start:end]
        change(name,block,'        if prepare_uv:\n'+indent(block,'    '))

    change('models/pipeline/ehm_pipeline.py','import lightning as L','import torch.nn as nn')
    change('models/pipeline/ehm_pipeline.py','class Ehm_Pipeline(L.LightningModule):','class Ehm_Pipeline(nn.Module):')
    change('models/pipeline/ehm_pipeline.py','from utils.pipeline_utils import *','# Unused offline plotting helpers are not needed for live inference.')
    change('models/smplx/pose_transformer.py','from utils.graphics import overlay_attention_on_image','# Unused offline attention renderer omitted.')
    for line in ['from .smplx_layer import SMPL_Layer','from .SMPLXV2 import SMPLX','from ..modules.ehm import EHM_v2 ',
                 'from pytorch3d.transforms import matrix_to_rotation_6d, matrix_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix']:
        change('models/smplx/smplx_head.py',line,'# Unused import: '+line)
    for name in ['models/modules/smplx/SMPLX.py','models/modules/ehm/EHM_v2.py']:
        change(name,'from pytorch3d.structures import Meshes','from mocap.pear.geometry_compat import Meshes')
    change('models/modules/flame/FLAME.py','from pytorch3d.io import load_obj','from mocap.pear.geometry_compat import load_obj')

    smplx='models/modules/smplx/SMPLX.py'
    ehm='models/modules/ehm/EHM_v2.py'
    change(smplx,'check_pose=True,add_teeth=False,uv_size=512):',
           'check_pose=True,add_teeth=False,uv_size=512, *, prepare_uv=True):')
    change(ehm,'flat_hand_mean=False, uv_size= 512):',
           'flat_hand_mean=False, uv_size= 512, *, prepare_uv=True):')
    change(ehm,'add_teeth=add_teeth, uv_size= uv_size)',
           'add_teeth=add_teeth, uv_size= uv_size, prepare_uv=prepare_uv)')
    guard_uv(smplx,'self.position_map=generate_position_map(')
    guard_uv(smplx,'uvmap_f_idx=get_uvmap_faces_index(',"self.register_buffer('uvmap_mask',")
    guard_uv(smplx,'self.get_head_idx_from_pos()')
    guard_uv(ehm,'self.get_head_idx_from_pos()')
