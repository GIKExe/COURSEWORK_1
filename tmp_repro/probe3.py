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

def stats(img, it):
    cnt=it["contour"].reshape(-1,2).astype(np.float64)
    ang=np.radians(it["angle_deg"]); ux,uy=np.cos(ang),np.sin(ang); vx,vy=-uy,ux
    x,y,w,h=it["bbox"]; pad=4
    x0,y0=max(0,x-pad),max(0,y-pad); x1,y1=min(img.shape[1],x+w+pad),min(img.shape[0],y+h+pad)
    crop=img[y0:y1,x0:x1]
    mask=np.zeros(crop.shape[:2],np.uint8)
    cv2.drawContours(mask,[(cnt-np.array([x0,y0])).astype(np.int32)],-1,255,cv2.FILLED)
    er=max(3,int(min(crop.shape[:2])*0.2)|1)
    inner=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(er,er)))
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur_k=max(3,int(min(crop.shape[:2])*0.06)*2+1)
    grayb=cv2.GaussianBlur(gray,(blur_k,0) if False else (blur_k,blur_k),0)
    ys,xs=np.mgrid[0:crop.shape[0],0:crop.shape[1]].astype(np.float32)
    t=xs*ux+ys*uy; s=xs*vx+ys*vy
    sel=inner>0
    sv=s[sel]; tv=t[sel]; gv=grayb[sel]
    sc=float(np.median(sv)); ss=float(np.percentile(sv,97)-np.percentile(sv,3))
    core=np.abs(sv-sc)<=max(3.0,ss*0.12)
    flank=np.abs(sv-sc)>max(3.0,ss*0.30)
    # flank restricted to the middle 70% of length (avoid end darkening)
    tbins=np.quantile(tv,[0.15,0.85])
    mid=(tv>=tbins[0])&(tv<=tbins[1])
    fmid=flank&mid
    depth=float(gv[fmid].mean()-gv[core&mid].mean()) if fmid.sum()>5 and (core&mid).sum()>5 else 0.0
    bins=np.quantile(tv,[0,.25,.5,.75,1])
    deltas=[]
    for k in range(4):
        m=core&(tv>=bins[k])&(tv<=bins[k+1]); f=flank&(tv>=bins[k])&(tv<=bins[k+1])
        if m.sum()<5 or f.sum()<5: continue
        deltas.append(float(gv[f].mean()-gv[m].mean()))
    dark_frac=sum(1 for d in deltas if d>1.0)/max(1,len(deltas))
    consistency=float(np.mean(deltas))/ (float(np.std(deltas))+2.0) if deltas else 0.0
    return dict(depth=depth,dark_frac=dark_frac,cons=consistency,nseg=len(deltas))

A=[stats(img1,a) for a,b in pairs]; B=[stats(img2,b) for a,b in pairs]
def pr(name,d,key):
    v=np.array([x[key] for x in d]); print(f"{name} {key}: p10={np.percentile(v,10):.2f} med={np.median(v):.2f} p90={np.percentile(v,90):.2f}")
for key in ("depth","dark_frac","cons"):
    pr("A",A,key); pr("B",B,key)
SA=np.array([max(0,x["depth"])*(0.5+x["dark_frac"]/2 + 0.5*max(0,min(1,x["cons"]))) for x in A])
SB=np.array([max(0,x["depth"])*(0.5+x["dark_frac"]/2 + 0.5*max(0,min(1,x["cons"]))) for x in B])
for thr in (0.5,1,1.5,2,2.5,3,4,5):
    print(f"S3>{thr}: FP={np.mean(SA>thr):.3f} TP={np.mean(SB>thr):.3f}")
