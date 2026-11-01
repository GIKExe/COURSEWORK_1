import sys; sys.path.insert(0,"/workspace/src")
import numpy as np, cv2
from core.segmentation import process_image
from core.params import get_default_params

p = get_default_params()
imgs={n:cv2.imread(f"/workspace/images/{n}.jpg") for n in ("2_1","2_1p")}
res={n:process_image(imgs[n],p)["items"] for n in imgs}
pairs=[]
for v in res["2_1"]:
    bd=1e9;best=None
    for v2 in res["2_1p"]:
        d=(v["center_x"]-v2["center_x"])**2+(v["center_y"]-v2["center_y"])**2
        if d<bd: bd,best=d,v2
    if bd<40**2: pairs.append((v,best))

def feats(img,it):
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
    bk=max(3,int(min(crop.shape[:2])*0.05)*2+1)
    gb=cv2.GaussianBlur(gray,(bk,bk),0)
    ys,xs=np.mgrid[0:crop.shape[0],0:crop.shape[1]].astype(np.float32)
    t=xs*ux+ys*uy; s=xs*vx+ys*vy
    sel=inner>0
    sv=s[sel]; tv=t[sel]; gv=gb[sel]
    sc=float(np.median(sv)); ss=float(np.percentile(sv,97)-np.percentile(sv,3))
    core=np.abs(sv-sc)<=max(3.0,ss*0.12)
    flank=np.abs(sv-sc)>max(3.0,ss*0.30)
    bins=np.quantile(tv,[0,.25,.5,.75,1])
    deltas=[]
    for k in range(4):
        m=core&(tv>=bins[k])&(tv<=bins[k+1]); f=flank&(tv>=bins[k])&(tv<=bins[k+1])
        if m.sum()<5 or f.sum()<5: continue
        deltas.append(float(gv[f].mean()-gv[m].mean()))
    dark_frac=sum(1 for d in deltas if d>1.0)/max(1,len(deltas))
    depth=float(np.mean(deltas)) if deltas else 0.0
    # central-band gradient (groove edges run along axis -> gradient ACROSS axis is high at groove sides)
    gx=cv2.Sobel(gb,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gb,cv2.CV_32F,0,1,ksize=3)
    g_across=np.abs(gx*(-uy)+gy*(ux))  # derivative along v direction? careful: v=(-uy,ux)
    band2=np.abs(s-sc)<=max(3.0,ss*0.30)
    band2&=sel
    gcore=float(np.median(g_across[band2])) if band2.sum()>10 else 0.0
    g_all=float(np.median(np.hypot(gx,gy)[sel]))
    return dict(depth=depth,dark_frac=dark_frac,gcore=gcore,g_all=g_all,nseg=len(deltas))

A=[feats(imgs["2_1"],a) for a,b in pairs]; B=[feats(imgs["2_1p"],b) for a,b in pairs]
def pr(n,d,k):
    v=np.array([x[k] for x in d]); print(f"{n} {k}: p10={np.percentile(v,10):.2f} med={np.median(v):.2f} p90={np.percentile(v,90):.2f}")
for k in ("depth","dark_frac","gcore","g_all"):
    pr("A",A,k); pr("B",B,k)

def sc(x, w_g=0.0):
    base = max(0.0,x["depth"]) * (x["dark_frac"] if x["nseg"]>=2 else 0.0)
    return base + w_g*x["gcore"]
for wg in (0.0,0.1,0.2,0.3):
    SA=np.array([sc(x,wg) for x in A]); SB=np.array([sc(x,wg) for x in B])
    best=None
    for thr in np.arange(0.5,8,0.25):
        fp=np.mean(SA>thr); tp=np.mean(SB>thr)
        acc=(fp+tp)/2  # error rate avg
        if best is None or acc<best[0]: best=(acc,thr,fp,tp)
    print(f"wg={wg}: best thr={best[1]:.2f} FP={best[2]:.3f} TP={best[3]:.3f} err={best[0]:.3f}")
# also check gcore alone thresholded
SA=np.array([x["gcore"] for x in A]); SB=np.array([x["gcore"] for x in B])
for thr in (6,8,10,12,14):
    print(f"gcore>{thr}: FP={np.mean(SA>thr):.3f} TP={np.mean(SB>thr):.3f}")
# combined rule: depth*darkfrac>thr OR gcore>thr2
SA2=np.array([max(0.0,x["depth"])*(x["dark_frac"] if x["nseg"]>=2 else 0) for x in A])
SB2=np.array([max(0.0,x["depth"])*(x["dark_frac"] if x["nseg"]>=2 else 0) for x in B])
GA=np.array([x["gcore"] for x in A]); GB=np.array([x["gcore"] for x in B])
for t1 in (1.0,1.5,2.0):
    for t2 in (9,10,11,12):
        fa=np.mean((SA2>t1)|(GA>t2)); fb=np.mean((SB2>t1)|(GB>t2))
        print(f"rule t1={t1} t2={t2}: FP={fa:.3f} TP={fb:.3f} err={(fa+1-fb)/2:.3f}")
