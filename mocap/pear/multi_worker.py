"""Resident multiview inputs; bounded capture buffers, parallel CPU decode/crops."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,time,uuid,re,shutil,zipfile
import numpy as np
from .input_transform import rotate_frame,input_rotations
from .multiview import Fusion, InvalidObservationDepth, observations, local_matrices, fk, CV
from .projection import Placement
from .camera_pose import resolve_cameras
from .fixed_rig import CalibrationFrames
from .temporal_filter import ParameterFilter
from ..paths import ROOT
from ..storage import atomic_json,sha256

def owned_folder(path):
    p=Path(path).resolve();root=(ROOT/'cache/pear/alignment').resolve()
    if not p.is_relative_to(root) or p==root:raise ValueError('Invalid PEAR alignment cache path')
    return p

def write_sample(path,image,cv):
    ok,encoded=cv.imencode('.png',image)
    if not ok:raise OSError('Cannot encode camera alignment sample')
    Path(path).write_bytes(encoded.tobytes())

def read_parameters(path):
    if path.stat().st_size>2*1024**2:raise ValueError('Alignment parameter file is too large')
    with zipfile.ZipFile(path) as z:
        entries=z.infolist()
        if len(entries)>64 or sum(p.file_size for p in entries)>2*1024**2:raise ValueError('Invalid alignment parameter archive')
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def prune_samples(keep=None):
    """At most four private jobs; never follow links or delete non-job paths."""
    root=ROOT/'cache/pear/alignment';root.mkdir(parents=True,exist_ok=True)
    jobs=[p for p in root.iterdir() if p.is_dir() and re.fullmatch(r'[0-9a-f]{32}',p.name) and not p.is_symlink() and not (getattr(p.lstat(),'st_file_attributes',0)&0x400)]
    ordered=sorted(jobs,key=lambda p:p.stat().st_mtime,reverse=True)
    retained=ordered[:3]
    protected=next((p for p in ordered if keep and p.resolve()==Path(keep).resolve()),None)
    if protected is not None and protected not in retained:retained=retained[:2]+[protected]
    for p in ordered:
        if p not in retained:owned_folder(p);shutil.rmtree(p)

def create_sample_folder(keep=None):
    prune_samples(keep)
    folder = ROOT/'cache/pear/alignment'/uuid.uuid4().hex
    folder.mkdir(parents=True)
    return folder

class Inputs:
    def __init__(self,sources,mode,height,pool,cv,streams=None):
        self.sources=sources;self.mode=mode;self.pool=pool;self.cv=cv;self.caps=[];self.last=[None]*len(sources)
        self.video_timing = []
        try:
            if streams is not None and len(streams)!=len(sources):raise ValueError('Shared camera count changed')
            for i,s in enumerate(sources):
                if mode=='camera':
                    from .camera_stream import open_camera
                    cap=open_camera(s,height,cv,streams[i] if streams is not None else None)
                else:
                    cap=cv.VideoCapture(s['path'],cv.CAP_FFMPEG,[cv.CAP_PROP_N_THREADS,1,cv.CAP_PROP_OPEN_TIMEOUT_MSEC,2000,cv.CAP_PROP_READ_TIMEOUT_MSEC,2000])
                    if not cap.isOpened():cap.release();raise ValueError('Cannot open camera video: '+Path(s['path']).name)
                self.caps.append(cap)
                if mode == 'video':
                    fps = cap.get(cv.CAP_PROP_FPS)
                    frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
                    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(frames) or frames < 1:
                        raise ValueError('Camera video has invalid timing.')
                    self.video_timing.append((fps, int(frames)))
        except Exception:self.close();raise
    def read(self,timestamp,rotation):
        angles=input_rotations({'rotations':rotation},len(self.caps)) if isinstance(rotation,(list,tuple)) else input_rotations({'rotation':rotation},len(self.caps))
        def one(i):
            cap=self.caps[i]
            if self.mode=='camera':
                item=cap.take()
                return None if item is None else (rotate_frame(item[0],angles[i]),item[1])
            fps, frames = self.video_timing[i]
            index = min(frames - 1, max(0, round(timestamp * fps)))
            last = self.last[i]
            if last is None or not last<index<=last+8:cap.set(self.cv.CAP_PROP_POS_FRAMES,index)
            else:
                for _ in range(index-last-1):cap.grab()
            ok,im=cap.read();self.last[i]=index
            if not ok:raise ValueError('Cannot decode camera '+str(i+1)+' at this time.')
            return rotate_frame(im,angles[i]),index/max(fps,1.)
        items=list(self.pool.map(one,range(len(self.caps))))
        if any(v is None for v in items):return None
        if self.mode=='camera':
            # Arrival-time synchronization is approximate: no claim of hardware genlock.
            times=[v[1] for v in items]
            if max(times)-min(times)>.08:return None
        return [v[0] for v in items],float(np.mean([v[1] for v in items]))
    def close(self):
        errors=[]
        for cap in self.caps:
            try:cap.close() if self.mode=='camera' else cap.release()
            except Exception as e:errors.append(e)
        self.caps=[]
        if errors:raise errors[0]

class MultiPreview:
    def __init__(self,model,cv,send,initial_rig=None,*,instant=False):
        self.initial_rig = initial_rig
        self.instant = instant
        if instant:
            from ..instant_live import InstantBatch as BatchRunner
        else:
            from .batch import BatchRunner
        self.pool=ThreadPoolExecutor(4,thread_name_prefix='live-cameras');self.batch=BatchRunner(model,self.pool)
        self.model=model;self.cv=cv;self.send=send;self.inputs=None;self.key=None;self.fusion=None;self.align_key=None
        self.camera_data=None;self.fusion_key=None;self.active_cameras=None
        self.inference_slots = ()
        self.thumbnail_key = None
        self.thumbnail_at = 0.
        self.reset_warning = ''
        from .foot_lock import LiveFootLock
        self.foot_lock=LiveFootLock();self.lock_options=None;self.position_anchor=None
        self.placement=Placement();self.filter=ParameterFilter();self.last_time=None;self.last_generation=None;self.pose_id=0;self.reference=None;self.reset_handled=0;self.references=CalibrationFrames()
    def close(self):
        try:
            if self.inputs:self.inputs.close();self.inputs=None
        finally:
            self.pool.shutdown(wait=True,cancel_futures=True)
            self.batch.close()
    def configure(self,request):
        sources=request['sources'];mode=request['mode']
        if not 1<=len(sources)<=32:raise ValueError('Live tracking needs 1–32 input slots.')
        if mode=='video':
            for entry in sources:
                st=Path(entry['path']).stat()
                if entry.get('file_stamp') and entry['file_stamp']!=[st.st_size,st.st_mtime_ns]:raise ValueError('A camera video changed. Open the videos again before tracking.')
        self.rotations=input_rotations(request,len(sources))
        key=json.dumps([mode,sources,request['resolution'],self.rotations,request.get('camera_streams')],sort_keys=True)
        if key!=self.key:
            if self.inputs:self.inputs.close();self.inputs=None
            self.inputs=Inputs(sources,mode,request['resolution'],self.pool,self.cv,request.get('camera_streams'));self.key=key
            self.inference_slots = ()
            self.reset_warning = ''
            self.fusion=None;self.align_key=None
            self.camera_data=None;self.fusion_key=None;self.active_cameras=None
            self.filter.reset();self.placement.reset();self.last_time=None
            self.fallback_slot = None
            if self.instant:
                self.model.rig.reset()
            elif self.initial_rig is None:
                self.model.rig.clear();self.reference=None;self.references.clear()
                self.reset_handled=int(request.get('bone_reset',0))
            else:
                self.reference, self.references, self.pose_id, self.reset_handled = self.initial_rig
                self.initial_rig = None
        calibration=request.get('calibration')
        if calibration!=self.align_key:
            self.fusion=None;self.align_key=None;self.camera_data=None;self.fusion_key=None;self.active_cameras=None
            if calibration:
                folder=owned_folder(calibration);data=json.loads((folder/'cameras.json').read_text('utf8'))
                if data['signature']!=request['signature'] or len(data['intrinsics'])!=len(sources):raise ValueError('Camera alignment no longer matches these inputs.')
                if data.get('rig_sha256')!=sha256(folder/'rig.npz'):raise ValueError('Camera alignment rig changed. Run Camera Align again.')
                if data.get('source_model', 'pear') != ('instant' if self.instant else 'pear'):
                    raise ValueError('Run Camera Align for the selected tracking model.')
                if not self.instant:
                    z=read_parameters(folder/'rig.npz')
                    body={k[5:]:v for k,v in z.items() if k.startswith('body_')}
                    flame={k[6:]:v for k,v in z.items() if k.startswith('flame_')}
                    self.send({'type':'status','text':'Restoring aligned fixed-rig calibration…','generation':request['generation']})
                    self.model.rig.calibrate(body,flame);self.reference=(body,flame)
                self.camera_data=data;self.align_key=calibration

    def configure_fusion(self, edits, images):
        sizes = [list(im.shape[1::-1]) for im in images]
        if self.camera_data and sizes != self.camera_data['sizes']:
            raise ValueError('Camera image dimensions changed. Run Camera Align again.')
        key = (edits, sizes)
        if key == self.fusion_key:
            return
        if self.camera_data or edits:
            data, active = resolve_cameras(self.camera_data, edits, sizes)
            if self.instant:
                from ..instant_alignment import assumed_intrinsics
                if self.camera_data is None:
                    data['intrinsics'] = [assumed_intrinsics(*size) for size in sizes]
                rig = self.model.rig
                self.fusion = Fusion(rig.fusion_rig, data, view=rig.fusion_view)
            else:
                self.fusion = Fusion(self.model.rig, data)
            self.active_cameras = active
        else:
            self.fusion = None
            self.active_cameras = None
        self.fusion_key = key
        self.foot_lock.reset()
        self.position_anchor = None
    def infer_views(self, images, slots, reset=False):
        if slots != self.inference_slots or reset:
            self.batch.reset(len(slots))
            self.inference_slots = slots
        packets = [None] * len(images)
        for index, packet in zip(slots, self.batch.run([images[i] for i in slots])):
            packets[index] = packet
        return packets

    def thumbnails(self,images):
        out=[]
        for im in images:
            h,w=im.shape[:2];s=min(128/w,72/h);thumb=np.zeros((72,128,3),np.uint8)
            small=self.cv.resize(im,(max(1,round(w*s)),max(1,round(h*s))))
            thumb[:small.shape[0],:small.shape[1]]=small[...,::-1];out.append(thumb)
        return np.stack(out)
    def prepare_alignment(self,request,images,packets,timestamp):
        folder=create_sample_folder(request.get('calibration'))
        n=len(images);samples=[];uv=[];boxes=[];sizes=[];paths=[];camera_relative=[]
        for sample in range(2):
            if sample:
                if request['mode']=='camera':
                    until=time.monotonic()+1.;fresh=None
                    while time.monotonic()<until:
                        item=self.inputs.read(timestamp,self.rotations)
                        if item:fresh=item
                    if fresh is None:raise ValueError('Webcams did not deliver synchronized fresh samples. Check connections.')
                    images,_=fresh;packets=self.batch.run(images)
                else:
                    end=float(request.get('duration',timestamp+2))-1/float(request.get('fps',30))
                    next_time=min(timestamp+2,end)
                    if next_time - timestamp < .5 / float(request.get('fps', 30)):
                        raise ValueError('Choose a playhead with at least one later frame for Camera Align.')
                    images,_=self.inputs.read(next_time,self.rotations);packets=self.batch.run(images)
            if sum(p is not None for p in packets)<min(2,n):raise ValueError('Camera Align needs the same person visible in the selected views. Choose another time.')
            obs=np.full((n,145,2),np.nan);bb=np.full((n,4),np.nan);relative=np.full((n,55,3),np.nan)
            valid = 0
            for c,(im,p) in enumerate(zip(images,packets)):
                path=folder/f'sample-{sample}-{c}.png'
                # Limit calibration source pixels; intrinsics are returned in original dimensions.
                h,w=im.shape[:2];scale=min(1.,1920/w,1080/h);small=self.cv.resize(im,(round(w*scale),round(h*scale))) if scale<1 else im
                write_sample(path,small,self.cv)
                paths.append(path.name)
                if p is not None and self.instant:
                    obs[c] = p['uv']
                    bb[c] = p['bbox']
                    relative[c] = p['camera_relative']
                    valid += 1
                elif p is not None:
                    try:
                        obs[c] = observations(p, self.model.rig)
                    except InvalidObservationDepth:
                        pass
                    else:
                        valid += 1
                        bb[c] = p['bbox']
                        local, _ = fk(local_matrices(p['body']), self.model.rig)
                        relative[c] = local[:55] @ (CV @ p['camera'][:3, :3]).T
                if sample==0:sizes.append([w,h])
                elif sizes[c]!=[w,h]:raise ValueError('Camera dimensions changed during alignment.')
            if valid < min(2, n):
                raise ValueError('Camera Align needs valid poses in the selected views. Choose another time.')
            uv.append(obs);boxes.append(bb);camera_relative.append(relative)
        if self.instant:
            rest = self.model.rig.fusion_rig.rest
            np.savez(folder/'rig.npz',rest=rest)
        else:
            body,flame=self.reference
            np.savez(folder/'rig.npz',**{'body_'+k:v for k,v in body.items()},**{'flame_'+k:v for k,v in flame.items()})
            rest = self.model.rig.rest
        np.savez(folder/'observations.npz',uv=np.asarray(uv),boxes=np.asarray(boxes),sizes=np.asarray(sizes),rest=rest,camera_relative=np.asarray(camera_relative))
        atomic_json(folder/'job.json',dict(version=1,source_model='instant' if self.instant else 'pear',
            signature=request['signature'],images=paths,count=n,
            rig_sha256=sha256(folder/'rig.npz'),observations_sha256=sha256(folder/'observations.npz')))
        self.send({'type':'alignment','phase':'prepared','folder':str(folder),'generation':request['generation']})
    def process(self,request):
        started=time.perf_counter();self.configure(request)
        timestamp=float(request.get('timeline_frame',0))/float(request.get('fps',30))
        item=self.inputs.read(timestamp,self.rotations)
        if item is None:
            if self.inputs.mode=='camera':self.send({'type':'status','text':'Waiting for synchronized camera frames (arrival skew ≤80 ms)…','generation':request['generation']})
            return
        images,source_time=item
        if self.inputs.mode == 'camera':
            # Match the single-stream worker. Wait for the capture before timing processing.
            started = time.perf_counter()
        if not request.get('align_job'):
            self.configure_fusion(request.get('camera_edits'), images)
        if request['mode']=='camera':timestamp=source_time
        continuous=self.last_time is not None and 0<timestamp-self.last_time<=.25 and self.last_generation==request['generation']
        if not continuous:
            self.filter.reset();self.placement.reset_filter();self.foot_lock.reset();self.position_anchor=None
            if self.instant:self.model.rig.reset_filter()
            if self.fusion:self.fusion.reset()
        reset_tracking = not continuous and self.last_time != timestamp
        self.last_time=timestamp;self.last_generation=request['generation']
        selected = min(max(0, int(request.get('preview', 0))), len(images) - 1)
        if request.get('align_job'):
            slots = tuple(range(len(images)))
        elif self.fusion is not None:
            slots = tuple(index for index, active in enumerate(self.active_cameras) if active)
        else:
            slots = (selected,)
        packets = self.infer_views(images, slots, reset_tracking)
        rig = self.model.rig
        available=[p for p in packets if p is not None]
        if rig.rest is None and available:
            rig.calibrate(available[0]['body'],available[0]['flame']);self.reference=(available[0]['body'],available[0]['flame'])
        if request.get('align_job'):
            if not available:raise ValueError('No person visible for Camera Align.')
            self.prepare_alignment(request,images,packets,timestamp);return 'prepared'
        reset = int(request.get('bone_reset', 0))
        if not self.instant and reset > self.reset_handled:
            self.reset_handled = reset
            self.reset_warning = ''
            reference = self.references.get(request.get('reset_pose'))
            if reference is None:
                self.reset_warning = 'Reset Bones: displayed pose expired. Pause and try again.'
            else:
                try:
                    rig.calibrate(*reference)
                except (ValueError, RuntimeError) as error:
                    self.reset_warning = f'Reset Bones: {error}'
                else:
                    self.reference = reference
                    self.filter.reset()
                    self.placement.reset_filter()
        arrays = {}
        smooth_rotations=bool(request['smoothing'] and request.get('smooth_rotations',True))
        smooth_position=bool(request['smoothing'] and request.get('smooth_position',True))
        if self.fusion:
            arrays = self.fusion.run(packets, timestamp, request['smoothing'], request['smoothing_strength'],
                smooth_rotations=request.get('smooth_rotations',True),smooth_position=request.get('smooth_position',True)) or {}
            view_count = self.fusion.available_views
        else:
            view_count = len(available)
        if self.fusion is None and packets[selected] is not None:
            p=packets[selected]
            if self.instant:
                if selected != self.fallback_slot:
                    self.fallback_slot = selected
                    rig.reset()
                arrays = rig.pose(p['state'], p['camera'], timestamp, smooth_rotations,
                                  smooth_position, request['smoothing_strength'])
            else:
                body=p['body'];self.filter.set_strength(request['smoothing_strength'])
                if smooth_rotations:body,_=self.filter.update(body,p['flame'],timestamp)
                else:self.filter.reset()
                v,j,r,f=rig.pose(body,p['camera']);v,j=self.placement.points(v,j,p['camera'],p['crop_bbox'],p['image_size'],smooth_position,timestamp,strength=request['smoothing_strength'])
                arrays=dict(bounds=np.ascontiguousarray(v[:,[0,2,1]]*[1,1,-1],dtype='float32'),joints=np.ascontiguousarray(j[:,[0,2,1]]*[1,1,-1],dtype='float32'),rotations=r,bone_facing=f)
        elif self.instant:
            self.fallback_slot = None
        foot_metrics={}
        options=(request.get('foot_lock',True),request.get('world_position',True),rig.revision,self.align_key)
        if options!=self.lock_options:self.foot_lock.reset();self.position_anchor=None;self.lock_options=options
        if arrays:
            if not request.get('world_position',True):
                from .projection import lock_position
                anchor=request.get('position_anchor')
                if anchor is not None:self.position_anchor=np.asarray(anchor,dtype='float32')
                if self.position_anchor is None:self.position_anchor=arrays['joints'][0].copy()
                arrays=lock_position(arrays,self.position_anchor)
            if not self.instant:
                arrays,foot_metrics=self.foot_lock.apply(arrays,rig,timestamp,
                    enabled=bool(request.get('foot_lock',True) and self.fusion is not None),move_root=request.get('world_position',True))
        else:self.foot_lock.reset()
        valid=bool(arrays);self.pose_id+=1
        if valid:
            arrays.update(rig.stream_reference())
        reference_packet=packets[selected] or (available[0] if available else None)
        if reference_packet and not self.instant:self.references.remember(self.pose_id,reference_packet['body'],reference_packet['flame'])
        im=images[selected];h,w=im.shape[:2];s=min(1,1280/w,720/h)
        if s<1:im=self.cv.resize(im,(round(w*s),round(h*s)))
        arrays['image'] = np.ascontiguousarray(im[..., ::-1])
        # The capture worker supplies webcam thumbnails. Video thumbnails use
        # a much lower rate than the selected full-size preview image.
        now = time.monotonic()
        thumb_key = request['generation'], selected
        if self.inputs.mode == 'video' and len(images) > 1 and (
                self.thumbnail_key != thumb_key or now - self.thumbnail_at >= .25):
            arrays['thumbnails'] = self.thumbnails(images)
            self.thumbnail_key, self.thumbnail_at = thumb_key, now
        self.send(dict(type='frame' if valid else 'missing',generation=request['generation'],request=request['request'],mode=request['mode'],
            preview=selected,frame=self.pose_id if request['mode']=='camera' else request.get('timeline_frame',0),timeline_frame=request.get('timeline_frame',0),pair_video=request.get('pair_video',False),
            requested_at=request.get('requested_at',time.monotonic()),timestamp=timestamp,render_mode='bones',pose_id=self.pose_id,bone_revision=rig.revision,
            bone_reset_handled=self.reset_handled,bone_reset_warning=self.reset_warning,
            ms=(time.perf_counter()-started)*1000,views=len(images),available_views=view_count,
            aligned=self.fusion is not None,smooth_rotations=smooth_rotations,smooth_position=smooth_position,
            foot_lock=foot_metrics,text='' if valid else 'No person visible in the selected inputs.'),**arrays)
