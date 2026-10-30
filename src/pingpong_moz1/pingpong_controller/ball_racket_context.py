"""Ball-observer-only racket FK, with an independent MuJoCo data/model copy.

The camera management pump reads an existing feedback cache; the camera
worker does FK. No policy/receiver callback performs this work. Device header
time is retained for diagnosis; unsynchronised headers use the original
receipt time, never the time at which a UI or camera packet is published.
"""
import hashlib
import pickle
import time
import xml.etree.ElementTree as ET

import numpy as np


def model_spec(controller):
    """Startup/profile-change only. The bytes never enter the per-frame packet."""
    model = getattr(controller, 'mj_model', None)
    data = getattr(controller, 'mj_data', None)
    if model is None or data is None:
        return None
    import mujoco as mj
    from pingpong_controller.rl_policy import DEPLOYMENT_FK_WORLD_Z_CORRECTION_M
    # The deployment model includes ~200 MB of rendering meshes. Extract only
    # the racket's ancestor chain: the same joint/body/site transforms, no
    # meshes, contacts, actuators or simulated dynamics. This also bounds IPC.
    chain=[];body=int(model.site_bodyid[controller._racket_site_id])
    while body>0:
        chain.append(body);body=int(model.body_parentid[body])
    chain.reverse()
    root=ET.Element('mujoco');ET.SubElement(root,'compiler',angle='radian')
    parent=ET.SubElement(root,'worldbody');joint_ids=[]
    numbers=lambda values:' '.join(format(float(x),'.17g') for x in values)
    for body in chain:
        if int(model.body_mocapid[body])>=0:
            raise ValueError('Mocap-driven robot bodies are not supported by camera racket FK')
        parent=ET.SubElement(parent,'body',name=f'body_{body}',
            pos=numbers(model.body_pos[body]),quat=numbers(model.body_quat[body]))
        ET.SubElement(parent,'inertial',pos='0 0 0',mass='1',diaginertia='1 1 1')
        for j in np.flatnonzero(model.jnt_bodyid==body):
            joint_ids.append(int(j));kind=int(model.jnt_type[j])
            attrs=dict(name=f'joint_{j}',type=('free','ball','slide','hinge')[kind])
            if kind!=0:attrs['pos']=numbers(model.jnt_pos[j])
            if kind>=2:
                attrs.update(axis=numbers(model.jnt_axis[j]),
                    ref=format(float(model.qpos0[model.jnt_qposadr[j]]),'.17g'),limited='false')
            ET.SubElement(parent,'joint',**attrs)
    site=int(controller._racket_site_id)
    ET.SubElement(parent,'site',name='camera_racket',pos=numbers(model.site_pos[site]),quat=numbers(model.site_quat[site]))
    compact=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    qpos=compact.qpos0.copy();joint_map={};address_map={}
    for j in joint_ids:
        nj=mj.mj_name2id(compact,mj.mjtObj.mjOBJ_JOINT,f'joint_{j}');joint_map[j]=nj
        old=int(model.jnt_qposadr[j]);new=int(compact.jnt_qposadr[nj]);width=(7,4,1,1)[int(model.jnt_type[j])]
        qpos[new:new+width]=data.qpos[old:old+width]
        address_map.update({old+k:new+k for k in range(width)})
    blob = pickle.dumps(compact, protocol=5)
    spec = dict(model=blob, qpos=qpos,
        joints=[joint_map[j] for j in controller._arm_jids], site=0,
        base=mj.mj_name2id(compact,mj.mjtObj.mjOBJ_BODY,f'body_{controller._base_body_id}'),
        base_addresses=[address_map.get(int(getattr(controller,k)),-1)
            for k in ('_base_x_qadr','_base_y_qadr','_base_yaw_qadr')],
        z_correction=float(DEPLOYMENT_FK_WORLD_Z_CORRECTION_M))
    if len(blob)>1024*1024:
        raise ValueError('Camera racket FK model exceeds the bounded configuration size')
    spec['id'] = hashlib.sha256(pickle.dumps(spec,protocol=5)).hexdigest()
    return spec


class FeedbackReader:
    """Read only immutable values already maintained by the ROS receiver."""
    def __init__(self, node):
        self.node = node
        self.previous_key = None
        self.previous = None

    def __call__(self):
        node = self.node
        # Do not wait behind the receiver/control lock. A missed read retains
        # the same source stamp and will expire naturally in the observer.
        if not node.lock.acquire(blocking=False):
            return self.previous
        try:
            q = node.latest_joint_q
            dq = node.latest_joint_dq
            receipt = node.latest_joint_stamp_monotonic_s
            wall = node.latest_joint_stamp_wall
            header = node.latest_joint_header_stamp_s
            sequence = node.latest_joint_source_sequence
            fault = node.latest_joint_source_fault
            key = (sequence, getattr(node,'latest_joint_sample_stamp_s',header))
            if q is None or receipt is None or fault:
                self.previous = None
                self.previous_key = key
                return None
            if key == self.previous_key:
                return self.previous
            q = np.asarray(q,float).copy()
            dq = None if dq is None else np.asarray(dq,float).copy()
        finally:
            node.lock.release()
        # Receipt monotonic and receiver ROS clocks are sampled together.
        # Conversion is frozen for this sample; repeated reads never refresh it.
        before = time.monotonic()
        now_ros = node.get_clock().now().nanoseconds*1e-9
        after = time.monotonic()
        if after-before > .002:
            return self.previous
        stamp = float(receipt)+now_ros-.5*(before+after)
        clock = 'receipt'
        if header is not None and wall is not None and 0 <= wall-header <= .050:
            stamp -= float(wall-header)
            clock = 'synchronised_header'
        packet = dict(q=q, dq=dq, stamp=stamp, sequence=int(sequence),
            header_stamp=header, receipt_monotonic=receipt, clock=clock)
        self.previous_key, self.previous = key, packet
        return packet


class RacketKinematics:
    """Worker-owned FK: no controller object, actuator or shared MjData."""
    def __init__(self, spec, sim_offset_m):
        import mujoco as mj
        self.mj = mj
        self.spec = spec
        # Only the trusted local startup channel supplies this model blob.
        self.model = pickle.loads(spec['model'])
        self.data = mj.MjData(self.model)
        self.data.qpos[:] = spec['qpos']
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in spec['joints']],int)
        self.vadr = np.array([self.model.jnt_dofadr[j] for j in spec['joints']],int)
        self.offset = np.asarray(sim_offset_m,float)
        self.jp = np.zeros((3,self.model.nv))
        self.jr = np.zeros_like(self.jp)
        self.key = None
        self.cached = (None,None)

    def evaluate(self, sample):
        if sample is None:
            self.key = None
            self.cached = (None,None)
            return self.cached
        key = (sample['sequence'],sample['stamp'])
        if key == self.key:
            return self.cached
        q = np.asarray(sample['q'],float)
        dq = np.asarray(sample.get('dq'),float)
        if q.shape != self.qadr.shape or not np.isfinite(q).all():
            return (None,None)
        self.data.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.model,self.data)
        self.mj.mj_comPos(self.model,self.data)
        self.mj.mj_jacSite(self.model,self.data,self.jp,self.jr,self.spec['site'])
        p = self.data.site_xpos[self.spec['site']].copy()
        normal = self.data.site_xmat[self.spec['site']].reshape(3,3)[:,2].copy()
        rotation = np.eye(3)
        if self.spec['base'] >= 0:
            addresses = self.spec['base_addresses']
            if min(addresses) >= 0:
                x,y,yaw = self.data.qpos[addresses]
            else:
                x,y = self.data.xpos[self.spec['base'],:2]
                r = self.data.xmat[self.spec['base']].reshape(3,3)
                yaw = np.arctan2(r[1,0],r[0,0])
            c,s = np.cos(yaw),np.sin(yaw)
            rotation = np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
            p = rotation@(p-np.array([x,y,0.]))
            p[2] += self.spec['z_correction']
        p -= self.offset
        normal = rotation@normal
        if normal[2] < 0:normal = -normal
        velocity_ok = dq.shape == q.shape and np.isfinite(dq).all()
        v = rotation@self.jp[:,self.vadr]@dq if velocity_ok else np.zeros(3)
        omega = rotation@self.jr[:,self.vadr]@dq if velocity_ok else np.zeros(3)
        if not np.isfinite([*p,*v,*normal,*omega,sample['stamp']]).all():
            return (None,None)
        velocity_ok = bool(velocity_ok and np.linalg.norm(v)<5. and np.linalg.norm(omega)<50.)
        if not velocity_ok:v=np.zeros(3);omega=np.zeros(3)
        state = (float(sample['stamp']),p*1000.,v*1000.)
        context = dict(version=1,source_sequence=sample['sequence'],
            stamp=sample['stamp'],clock=sample['clock'],header_stamp=sample['header_stamp'],
            normal=normal,angular_velocity=omega,velocity_valid=velocity_ok,model_id=self.spec['id'])
        self.key, self.cached = key,(state,context)
        return self.cached
