"""Bounded IR-constrained sphere fit for incomplete stereo patches.

Only real depth pixels enter the fit. A positive curvature check rejects a
bright racket plane; the downstream observation model still checks identity.
"""
import numpy as np


def local_surface(ir,depth,u,v,radius,fx,fy,cx,cy,scale):
    if depth is None or not np.isfinite([u,v,radius]).all() or not 8<radius<90:return None
    h,w=depth.shape
    x0,x1=max(0,int(u-radius)),min(w,int(u+radius)+2)
    y0,y1=max(0,int(v-radius)),min(h,int(v+radius)+2)
    if x1<=x0 or y1<=y0:return None
    yy,xx=np.mgrid[y0:y1,x0:x1];disk=(xx-u)**2+(yy-v)**2<(.75*radius)**2
    zz=depth[y0:y1,x0:x1].astype(float)*scale
    good=disk&(zz>.15)&(zz<1.)
    if np.count_nonzero(good)<80 or np.count_nonzero(good)/max(1,disk.sum())<.25:return None
    values=zz[good];median=np.median(values)
    good&=abs(zz-median)<.015
    ys,xs=yy[good],xx[good];z=zz[good]
    if len(z)<80 or np.ptp(xs)<.75*radius or np.ptp(ys)<.75*radius:return None
    if np.count_nonzero(ir[y0:y1,x0:x1][good]>80)<20:return None
    take=np.linspace(0,len(z)-1,min(128,len(z))).astype(int)
    xs,ys,z=xs[take],ys[take],z[take]
    points=np.column_stack(((xs-cx)*z/fx,(ys-cy)*z/fy,z))
    ray=np.array([(u-cx)/fx,(v-cy)/fy,1.]);direction=ray/np.linalg.norm(ray)
    distance=(median+.016)*np.linalg.norm(ray)
    for _ in range(6):
        diff=distance*direction-points;norm=np.linalg.norm(diff,axis=1)
        jacobian=diff@direction/np.maximum(norm,1e-6);residual=norm-.020
        weight=np.minimum(1.,.002/np.maximum(abs(residual),1e-6))
        step=float((weight*jacobian)@residual/(np.sum(weight*jacobian**2)+1e-6))
        distance-=np.clip(step,-.010,.010)
    center=distance*direction;residual=np.linalg.norm(center-points,axis=1)-.020
    rms=float(np.sqrt(np.mean(residual**2)))
    expected=np.sqrt(fx*fy)*.020/center[2]
    if rms>.002 or center[2]<median+.004 or abs(radius-expected)/expected>.25:return None
    # A flat racket patch can have a small sphere residual over a narrow cap.
    # Require positive spherical curvature after allowing a tilted plane.
    axial=points@direction;radial=points-axial[:,None]*direction
    rho2=np.sum(radial**2,axis=1)
    if np.any(rho2>=.020**2):return None
    sag=.020-np.sqrt(.020**2-rho2)
    design=np.column_stack((np.ones(len(points)),radial[:,:2],sag))
    curvature=np.linalg.lstsq(design,axial,rcond=None)[0][3]
    if not .4<curvature<1.8:return None
    return dict(center_m=center,fit_rms_mm=rms*1000.,support_fraction=float(good.sum()/max(1,disk.sum())),support_pixels=int(good.sum()),curvature=float(curvature))
