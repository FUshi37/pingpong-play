"""Bounded causal contact-event refinement; no truth or future samples."""
from collections import deque
import numpy as np


class ContactRefinement:
    def __init__(self):
        self.samples = deque(maxlen=8)
        self.anchor = None
        self.confirmed = False
        self.last_time = None
        self.conservative = True

    def refine(self, contact, model, raw, position, valid, t):
        plan = contact.plan
        if valid and (self.last_time is None or t > self.last_time+1e-6):
            self.samples.append((t, position.copy(), .003))
        self.last_time = t
        if plan is None or t-plan['time'] > .060:
            self.anchor = None
            self.confirmed = False
            return
        if self.conservative and not self.confirmed:
            dt=t-plan['time']
            if dt<0:
                return
            planned=plan['position']+dt*plan['velocity']+.5*dt*dt*model.gravity
            if not valid or np.linalg.norm(position-planned)<.012:
                return
        if self.anchor is not None and abs(plan['time']-self.anchor['time']) > .100:
            self.anchor = None
            self.confirmed = False
        if not self.confirmed:
            # Freeze an incoming arc only in the small impact window. Its
            # trajectory is independent of the current unaccepted range.
            if not valid or not -.022 <= t-plan['time'] <= .035:
                return
            prior = model._propagated(t)
            if prior is None or prior[0][5] >= -.4:
                return
            d = plan['time']-t
            self.anchor = dict(time=plan['time'],
                position=prior[0][:3]+d*prior[0][3:]+.5*d*d*model.gravity,
                incoming=prior[0][3:]+d*model.gravity,
                outgoing=plan['velocity'].copy())
        a = self.anchor
        if a is None or not valid or t-a['time'] > .045:
            return
        current_dt = t-a['time']
        free = a['position']+current_dt*a['incoming']+.5*current_dt**2*model.gravity
        if not self.confirmed and (position[2]-free[2] < .0045
                or np.linalg.norm(position[:2]-free[:2]) > .020):
            return
        times = np.array([x[0] for x in self.samples])
        points = np.array([x[1] for x in self.samples])
        candidates = a['time']+np.linspace(-.014,.010,13)
        best = None
        for tc in candidates:
            if tc > t-.001:
                continue
            dt = times-tc
            mask = (dt >= .001) & (dt <= .055)
            if not np.any(mask):
                continue
            dt = dt[mask];p = points[mask]
            delta = tc-a['time']
            pc = a['position']+delta*a['incoming']+.5*delta**2*model.gravity
            vp = a['outgoing']+delta*model.gravity
            target = p-pc-.5*dt[:,None]**2*model.gravity
            # The outgoing velocity is a noisy physical prior. Contact time
            # and the shared contact point must move together along ingress.
            strength=(.003/.30)**2
            v=(np.sum(dt[:,None]*target,axis=0)+strength*vp)/(dt@dt+strength)
            residual=target-dt[:,None]*v
            if v[2]<.25 or v[2]>4. or np.max(np.linalg.norm(residual,axis=1))>.012:
                continue
            score=np.sum(residual**2)/.003**2+np.sum((v-vp)**2)/.30**2+((tc-a['time'])/.008)**2
            if best is None or score<best[0]:
                best=(score,tc,pc,v,mask)
        if best is None:
            return
        score,tc,pc,v,mask=best
        dd=times[mask]-a['time']
        free_points=a['position']+dd[:,None]*a['incoming']+.5*dd[:,None]**2*model.gravity
        free_cost=np.sum((points[mask]-free_points)**2)/.003**2
        if not self.confirmed and free_cost-score<2.:
            return
        old_time=plan['time']
        contact.plan=dict(plan,time=tc,position=pc,velocity=v,refined=True)
        if contact.seeded==old_time:
            contact.seeded=tc
            model.contact_velocity_prior=(tc,v.copy())
        self.confirmed=True
        raw['observation_contact_refined']=True
        raw['observation_contact_time_correction_s']=tc-a['time']
        raw['observation_contact_real_samples']=int(mask.sum())

