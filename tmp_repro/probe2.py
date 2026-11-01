import sys; sys.path.insert(0,"/workspace/src")
import numpy as np, cv2
from core.segmentation import process_image
from core.params import get_default_params

p = get_default_params()
img1 = cv2.imread("/workspace/images/2_1.jpg"); img2 = cv2.imread("/workspace/images/2_1p.jpg")
r1 = process_image(img1, p); r2 = process_image(img2, p)
pairs=[]
for v in r1["items"]:
    bd=1e9; best=None
    for v2 in r2["items"]:
        d=(v["center_x"]-v2["center_x"])**2+(v["center_y"]-v2["center_y"])**2
        if d<bd: bd,best=d,v2
    if bd<40**2: pairs.append((v,best))
print("pairs:", len(pairs))

def stats(img, it):
    cnt = it["contour"].reshape(-1,2).astype(np.float64)
    ang=np.radians(it["angle_deg"]); ux,uy=np.cos(ang),np.sin(ang); vx,vy=-uy,ux
    x,y,w,h=it["bbox"]; pad=4
    x0,y0=max(0,x-pad),max(0,y-pad); x1,y1=min(img.shape[1],x+w+pad),min(img.shape[0],y+h+pad)
    crop=img[y0:y1,x0:x1]
    mask=np.zeros(crop.shape[:2],np.uint8)
    cv2.drawContours(mask,[(cnt-np.array([x0,y0])).astype(np.int32)],-1,255,cv2.FILLED)
    er=max(3,int(min(crop.shape[:2])*0.2)|1)
    inner=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(er,er)))
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    ys,xs=np.mgrid[0:crop.shape[0],0:crop.shape[1]].astype(np.float32)
    t=xs*ux+ys*uy; s=xs*vx+ys*vy
    sel=inner>0
    sv=s[sel]; tv=t[sel]; gv=gray[sel]
    sc=float(np.median(sv)); ss=float(np.percentile(sv,97)-np.percentile(sv,3))
    ts_span=float(np.percentile(tv,97)-np.percentile(tv,3))
    core=np.abs(sv-sc)<=max(3.0,ss*0.12)
    flank=np.abs(sv-sc)>max(3.0,ss*0.30)
    depth=float(gv[flank].mean()-gv[core].mean())
    bins=np.quantile(tv,[0,.25,.5,.75,1])
    seg_dark=0;nseg=0;deltas=[]
    for k in range(4):
        m=core&(tv>=bins[k])&(tv<=bins[k+1]); f=flank&(tv>=bins[k])&(tv<=bins[k+1])
        if m.sum()<5 or f.sum()<5: continue
        nseg+=1; dd=float(gv[f].mean()-gv[m].mean()); deltas.append(dd)
        if dd>1.0: seg_dark+=1
    dark_frac=seg_dark/max(1,nseg)
    # gradient along the core stripe (perpendicular direction derivative)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    g_along_core=float(np.abs(gx*ux+gy*uy)[sel&(np.abs(s-sc)<=max(3.0,ss*0.25))].mean())
    g_across_all=float(np.hypot(gx,gy)[sel].mean())
    return dict(depth=depth,dark_frac=dark_frac,gac=g_across_all,gcore=g_along_core,
                mean_delta=float(np.mean(deltas)) if deltas else 0.0)

A=[stats(img1,a) for a,b in pairs]; B=[stats(img2,b) for a,b in pairs]
def pr(name,d,key):
    v=np.array([x[key] for x in d]); print(f"{name} {key}: p10={np.percentile(v,10):.2f} med={np.median(v):.2f} p90={np.percentile(v,90):.2f}")
for key in ("depth","dark_frac","gac","gcore","mean_delta"):
    pr("A(no-gr)",A,key); pr("B(gr)",B,key)
# combined score candidates
def score(x): return x["mean_delta"]*x["dark_frac"]
SA=np.array([score(x) for x in A]); SB=np.array([score(x) for x in B])
for thr in (0.5,1,1.5,2,2.5,3,4):
    print(f"thr={thr}: FP={np.mean(SA>=thr):.3f} TP={np.mean(SB>=thr):.3f}")
G=np.array([x["gac"] for x in A]), np.array([x["gac"] for x in B])
for thr in (10,15,20,25):
    print(f"gac>{thr}: FP={np.mean(G[0]>thr):.3f} TP={np.mean(G[1]>thr):.3f}")
S2=np.array([x["mean_delta"]*x["dark_frac"] + 0.15*x["gac"] for x in A]), np.array([x["mean_delta"]*x["dark_frac"] + 0.15*x["gac"] for x in B])
for thr in (2,3,4,5,6,7,8):
    print(f"S2>{thr}: FP={np.mean(S2[0]>thr):.3f} TP={np.mean(S2[1]>thr):.3f}")
