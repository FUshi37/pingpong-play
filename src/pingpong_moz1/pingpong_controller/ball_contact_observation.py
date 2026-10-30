"""Causal, bounded contact prior using already received racket poses.

This is ball observation math only. It neither changes racket feedback nor
commands a bounce. Trusted post-contact ranges must confirm the new segment.
"""
from collections import deque
import numpy as np
from pingpong_controller.ball_contact_refinement import ContactRefinement


class BallContactObservation:
    def __init__(self):
        self.rackets=deque(maxlen=6)
        self.plan=None
        self.seeded=None
        self.motion=None
        self.context=None
        self.refinement=ContactRefinement()

    def racket_motion(self,state,t):
        if state is None:
            return None
        rt,p,v=state;rt=float(rt);p=np.asarray(p,float)/1000.;v=np.asarray(v,float)/1000.
        if p.shape!=(3,) or v.shape!=(3,) or not np.isfinite([rt,*p,*v]).all() or abs(t-rt)>.060:
            return None
        if not self.rackets or rt>self.rackets[-1][0]+1e-6:
            self.rackets.append((rt,p.copy()))
        points=[(tt,pp) for tt,pp in self.rackets if rt-tt<=.055]
        acceleration=np.zeros(3)
        context=self.context or {}
        source_velocity=bool(context.get('velocity_valid')
            and abs(float(context.get('stamp',-1e30))-rt)<1e-6)
        if not source_velocity and len(points)>=2 and points[-1][0]-points[0][0]>.005:
            times=np.array([tt-rt for tt,_ in points]);positions=np.array([pp for _,pp in points])
            # Feedback can repeat a pose or pause for several camera frames.
            # A short linear fit is less sensitive to that timing jitter than
            # differentiating the reported velocity or extrapolating a parabola.
            design=np.column_stack((np.ones(len(times)),times))
            fit=np.linalg.lstsq(design,positions,rcond=None)[0]
            v=fit[1]
        age=t-rt
        return p+age*v+.5*age**2*acceleration,v+age*acceleration,acceleration

    def prepare(self,model,raw,position,valid,t,racket_state):
        racket=self.racket_motion(racket_state,t)
        self.motion=racket
        prior=model._propagated(t)
        if self.plan is not None and t-self.plan['time']>.080:
            self.plan=None
        if racket is not None and prior is not None and model.velocity_valid and not model.supported:
            p,v=prior[0][:3],prior[0][3:];rp,rv,ra=racket;relative=v-rv
            context=self.context or {}
            normal=np.asarray(context.get('normal',[0.,0.,1.]),float)
            if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<.5:
                normal=np.array([0.,0.,1.])
            normal=normal/np.linalg.norm(normal)
            height=float(raw.get('ball_contact_height_m',.0082))
            restitution=float(raw.get('ball_contact_restitution',.67))
            separation=p-rp;normal_distance=float(separation@normal)
            normal_speed=float(relative@normal)
            if (normal[2]>.5 and normal_speed<-.4 and height<=normal_distance<=.25
                    and (self.plan is None or t<self.plan['time'])):
                distance=normal_distance-height;az=float((model.gravity-ra)@normal)
                if abs(az)<1e-6:
                    horizon=-distance/normal_speed
                else:
                    discriminant=normal_speed**2-2*az*distance
                    roots=((-normal_speed+np.sqrt(discriminant))/az,(-normal_speed-np.sqrt(discriminant))/az) if discriminant>=0 else ()
                    roots=[x for x in roots if x>=0]
                    horizon=min(roots) if roots else 1.
                if 0<=horizon<=.080 and np.linalg.norm(p[:2]-rp[:2]+horizon*relative[:2])<=.080:
                    pc=p+horizon*v+.5*horizon**2*model.gravity;vc=v+horizon*model.gravity
                    rvc=rv+horizon*ra
                    # Relative normal restitution; tangent velocity is retained
                    # until real post-contact ranges identify a change.
                    vp=vc-(1.+restitution)*float((vc-rvc)@normal)*normal
                    if .25<vp[2]<4.:
                        self.plan=dict(time=t+horizon,position=pc,velocity=vp)
        self.refinement.refine(self,model,raw,position,valid,t)
        plan=self.plan
        if plan is None:
            return
        dt=t-plan['time'];pp=plan['position']+dt*plan['velocity']+.5*dt**2*model.gravity
        vv=plan['velocity']+dt*model.gravity
        consistent=not valid or np.linalg.norm(position-pp)<.025
        if 0<=dt<=.050 and consistent:
            if self.seeded!=plan['time']:
                old_range=model.last_range_time
                model._new_segment(plan['time'])
                model.state=np.r_[pp,vv];model.covariance=np.diag([.004**2]*3+[.20**2]*3)
                model.state_time=t;model.velocity_valid=True;model.last_range_time=old_range
                self.seeded=plan['time']
                model.contact_velocity_prior=(plan['time'],plan['velocity'].copy())
            # The earlier prior has already started this flight segment.
            # Repeated legacy impact flags must not erase its real samples.
            raw['flight_segment_reset']=False;raw['contact_pending']=False
            if raw.get('observation_state')=='impact_confirming':
                raw['observation_state']='contact_prior_update'

    def publish(self,model,raw,t):
        raw['observation_bounce_horizon_s']=None
        raw['observation_contact_confirm_deadline_s']=None
        plan=self.plan
        if plan is None:
            return
        dt=t-plan['time']
        if dt<=0:
            raw['observation_bounce_horizon_s']=-dt
            for i,k in enumerate('xyz'):
                raw['observation_bounce_position_'+k+'_m']=float(plan['position'][i])
                raw['observation_bounce_velocity_'+k+'_m_s']=float(plan['velocity'][i])
        elif self.seeded!=plan['time']:
            # A rejected model prior must yield to real post-contact evidence.
            real=[s for s in model.history if s[0]>=plan['time']-.010]
            confirmed=(model.velocity_valid and model.fit_rmse is not None
                and len(real)>=2 and real[-1][0]-real[0][0]>=.009
                and t-real[-1][0]<.020 and model.state is not None and model.state[5]>.25)
            if confirmed:self.plan=None
            else:raw['observation_contact_horizon_s']=0.
        elif model.fit_rmse is None:
            raw['observation_contact_confirm_deadline_s']=.050-dt
