"""Fit a known-radius ball to original stereo pixels at image boundaries.

Only spatial observations enter this bounded fallback. Its result still needs
the shared trajectory association; a tiny low-residual stereo patch is rejected.
"""
import cv2
import numpy as np


def partial_ball_candidates(ir,depth,*,fx,fy,cx,cy,depth_scale,radius_m=.020):
    if depth is None or depth.shape!=ir.shape:
        return []
    h,w=ir.shape
    candidates=[]
    for y0,y1 in ((0,min(80,h)),(max(0,h-80),h)):
        d=depth[y0:y1].astype(float)*depth_scale
        mask=((d>.15)&(d<.9)).astype(np.uint8)
        _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for k in (np.argsort(stats[1:,4])[::-1][:8]+1):
            x,y,bw,bh,area=stats[k]
            if area<200 or not 5<=bw<=120 or not 4<=bh<=80:
                continue
            if not ((y0==0 and y<=1) or (y1==h and y+bh>=y1-y0-1)):
                continue
            yy,xx=np.nonzero(labels[y:y+bh,x:x+bw]==k);xx+=x;yy+=y
            z=d[yy,xx]
            if np.count_nonzero(ir[y0+yy,xx]>80)<8:
                continue
            median=np.median(z);keep=abs(z-median)<.020
            xx,yy,z=xx[keep],yy[keep],z[keep]
            if len(z)<200:
                continue
            take=np.linspace(0,len(z)-1,min(160,len(z))).astype(int)
            xx,yy,z=xx[take],yy[take]+y0,z[take]
            points=np.column_stack(((xx-cx)*z/fx,(yy-cy)*z/fy,z))
            center=np.array([np.median(points[:,0]),np.median(points[:,1]),np.median(z)+.85*radius_m])
            initial=center.copy()
            for _ in range(6):
                diff=center-points;distance=np.linalg.norm(diff,axis=1)
                jacobian=diff/np.maximum(distance[:,None],1e-6);residual=distance-radius_m
                weight=np.minimum(1.,.003/np.maximum(abs(residual),1e-6))
                matrix=jacobian.T@(weight[:,None]*jacobian)+.01*np.eye(3)
                update=np.linalg.solve(matrix,jacobian.T@(weight*residual))
                center-=np.clip(update,-.010,.010)
            residual=np.linalg.norm(center-points,axis=1)-radius_m
            rms=float(np.sqrt(np.mean(residual**2)))
            if rms>.003 or center[2]<median or np.linalg.norm(center-initial)>.045:
                continue
            u,v=fx*center[0]/center[2]+cx,fy*center[1]/center[2]+cy
            radius=np.sqrt(fx*fy)*radius_m/center[2]
            if min(u,w-1-u)<radius or not -radius<v<h+radius:
                continue
            # Require dense original stereo over the *visible* ball disk.
            x0,x1=max(0,int(u-radius)),min(w,int(u+radius)+2)
            z0,z1=max(0,int(v-radius)),min(h,int(v+radius)+2)
            gy,gx=np.ogrid[z0:z1,x0:x1]
            disk=(gx-u)**2+(gy-v)**2<(.85*radius)**2
            values=depth[z0:z1,x0:x1][disk]*depth_scale
            support=np.count_nonzero(abs(values-median)<.020)/max(1,len(values))
            if support<.60:
                continue
            candidates.append(dict(cx=float(u),cy=float(v),radius=float(radius),center_m=center,
                support_fraction=float(support),support_pixels=int(area),fit_rms_mm=rms*1000.))
    return candidates
