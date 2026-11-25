"""Fixed-rig, causal multiview fusion. No mesh solve or per-frame calibration."""
import numpy as np
from scipy.spatial.transform import Rotation
from .bone_roll import JOINTS,TIPS
from .camera_pose import DISPLAY
from .temporal_filter import ParameterFilter,AdaptiveFilter,strength_scale

CV=np.diag([-1.,-1.,1.])
BODY_IDS=np.array([16,17,18,19,20,21,1,2,4,5,7,8])


class InvalidObservationDepth(ValueError):
    pass

def local_matrices(body):
    return np.concatenate([body['global_pose'].reshape(1,3,3),body['body_pose'].reshape(21,3,3),
        np.tile(np.eye(3),(3,1,1)),body['left_hand_pose'].reshape(15,3,3),body['right_hand_pose'].reshape(15,3,3)])

def fk(local,rig):
    rotations=local.copy();joints=np.zeros((145,3));rest=rig.rest
    for k in range(1,55):
        p=int(rig.native_parents[k]);rotations[k]=rotations[p]@local[k]
        joints[k]=joints[p]+rotations[p]@(rest[k]-rest[p])
    for k,offset in rig.tip_offsets.items():joints[TIPS[k]]=joints[k]+rotations[k]@offset
    return joints,rotations

def observations(packet, rig, joints=None):
    if joints is None:
        joints, _ = fk(local_matrices(packet['body']), rig)
    cam = packet['camera']
    q = (joints + rig.rest[0]) @ cam[:3, :3].T + cam[:3, 3]
    crop = packet['crop_bbox']
    if not np.isfinite(q[:55]).all() or np.any(q[:55, 2] <= .01):
        raise InvalidObservationDepth('Invalid PEAR crop-camera depth')
    return crop[:2]+crop[2:]/2-12*crop[2:]*q[:,:2]/q[:,2:]

def pear_view(packet, rig, world_from_camera):
    local = local_matrices(packet['body'])
    joints, _ = fk(local, rig)
    uv = observations(packet, rig, joints)
    rotation = world_from_camera @ CV @ packet['camera'][:3, :3]
    local[0] = rotation @ local[0]
    return local, uv, joints @ rotation.T


class Fusion:
    def __init__(self,rig,cameras,*,view=pear_view):
        self.view=view
        self.rig=rig;self.K=np.asarray(cameras['intrinsics']);self.E=np.asarray(cameras['extrinsics'])
        n=len(self.K)
        if self.K.shape!=(n,3,3) or self.E.shape!=(n,3,4) or not 1<=n<=32 or not np.isfinite(self.K).all() or not np.isfinite(self.E).all():raise ValueError('Invalid camera alignment arrays')
        rr=self.E[:,:,:3]
        if np.any(self.K[:,[0,1],[0,1]]<=0) or np.max(abs(rr.swapaxes(-1,-2)@rr-np.eye(3)))>1e-4 or np.min(np.linalg.det(rr))<.999:raise ValueError('Invalid camera alignment rotations or intrinsics')
        self.P = self.K @ self.E
        self.direction_weights = np.zeros((55, 145))
        for joint in range(55):
            children = np.flatnonzero(rig.native_parents == joint)
            if len(children):
                self.direction_weights[joint, children] = 1. / len(children)
                self.direction_weights[joint, joint] -= 1.
            else:
                self.direction_weights[joint, max(0, rig.native_parents[joint])] = -1.
                self.direction_weights[joint, joint] += 1.
        self.filter=ParameterFilter();self.root_filter=AdaptiveFilter(1.2,4.,vector=True);self.reset()
    def reset(self):
        self.filter.reset();self.root_filter.reset();self.weights=None;self.time=None
        self.available_views = 0
    def quality(self,c,j,uv,packet):
        rotation = self.E[c, :, :3]
        box, size = packet['bbox'], packet['image_size']
        scale = np.clip(np.sqrt(max(box[3] - box[1], 1) / 420), .45, 1.5)
        camera_joints = j[:55] @ rotation.T
        torso = (uv[3] + uv[9]) / 2
        radius = max(12, np.linalg.norm(uv[16] - uv[17]) * .38)
        directions = (self.direction_weights @ j) @ rotation.T
        depth = directions[:, 2] / np.maximum(np.linalg.norm(directions, axis=1), 1e-6)
        observable = np.sqrt(np.maximum(0., 1. - depth**2))
        border = np.minimum(uv[:55], size - uv[:55]).min(axis=1)
        inside = np.clip(border / 40, .05, 1.)
        occlusion_joints = np.array([18, 19, 20, 21, 4, 5, 7, 8])
        occluded = ((np.linalg.norm(uv[occlusion_joints] - torso, axis=1) < radius)
                    & (camera_joints[occlusion_joints, 2] > camera_joints[9, 2] + .08))
        weights = scale * (.15 + .85 * observable) * inside
        weights[occlusion_joints[occluded]] *= .25
        weights[25:40] = weights[20]
        weights[40:55] = weights[21]
        return weights
    def run(self,packets,timestamp,smoothing=True,strength=50,smooth_rotations=True,smooth_position=True):
        available = []
        obs={};quality=np.zeros((len(packets),55));locals=[]
        for c, packet in enumerate(packets):
            if packet is None:
                continue
            try:
                local, obs[c], joints = self.view(packet, self.rig, self.E[c, :, :3].T)
            except InvalidObservationDepth:
                continue
            available.append(c)
            locals.append(local)
            quality[c] = self.quality(c, joints, obs[c], packet)
        if not available:
            self.reset()
            return None
        self.available_views = len(available)
        weights=quality/np.maximum(quality.sum(0),1e-8);dt=0 if self.time is None else timestamp-self.time
        if self.weights is not None and 0<dt<=.25:weights=self.weights+(1-np.exp(-dt/.1))*(weights-self.weights)
        weights[[i for i in range(len(packets)) if i not in available]]=0
        weights/=np.maximum(weights.sum(0),1e-8);self.weights=weights;self.time=timestamp
        L = locals[0]
        if len(available) > 1:
            A = np.asarray(locals)
            for _ in range(3):
                delta = Rotation.from_matrix((L.swapaxes(-1, -2)[None] @ A).reshape(-1, 3, 3))
                delta = delta.as_rotvec().reshape(len(available), 55, 3)
                L = L @ Rotation.from_rotvec((weights[available, :, None] * delta).sum(0)).as_matrix()
        if smoothing and smooth_rotations:
            self.filter.set_strength(strength)
            b=dict(global_pose=L[:1][None],body_pose=L[1:22][None],left_hand_pose=L[25:40][None],right_hand_pose=L[40:55][None])
            b,_=self.filter.update(b,{},timestamp);L=local_matrices(b)
        else:self.filter.reset()
        j,r=fk(L,self.rig);aa=[];bb=[]
        for c in available:
            p = self.P[c]
            equations = p[:2] - obs[c][BODY_IDS, :, None] * p[2]
            mat, offset = equations[..., :3], equations[..., 3]
            weights = np.sqrt(np.maximum(.02, quality[c, BODY_IDS]))
            aa.append((mat * weights[:, None, None]).reshape(-1, 3))
            targets = -(mat @ j[BODY_IDS, :, None])[..., 0] - offset
            bb.append((targets * weights[:, None]).ravel())
        a=np.concatenate(aa);b=np.concatenate(bb);root,_,rank,_=np.linalg.lstsq(a,b,rcond=None)
        if rank<3 or not np.isfinite(root).all() or np.linalg.norm(root)>1000:raise ValueError('Camera geometry cannot determine a stable position. Run Camera Align again.')
        if smoothing and smooth_position:
            scale=strength_scale(strength);self.root_filter.min_cutoff=1.2/scale;self.root_filter.beta=4/scale;self.root_filter.max_cutoff=8/scale
            root=self.root_filter.update(root,dt)
        else:self.root_filter.reset()
        j+=root
        return dict(joints=np.ascontiguousarray(j@DISPLAY.T,dtype='float32'),
            bounds=np.ascontiguousarray(j[np.r_[JOINTS,list(TIPS.values())]]@DISPLAY.T,dtype='float32'),
            rotations=np.ascontiguousarray(DISPLAY@r,dtype='float32'),
            bone_facing=np.ascontiguousarray(DISPLAY@(r[JOINTS]@self.rig.basis),dtype='float32'))
