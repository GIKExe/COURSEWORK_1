import sys; sys.path.insert(0,"/workspace/src")
import numpy as np, cv2
from core.segmentation import process_image
from core.params import get_default_params

p = get_default_params()
imgs = {}
for n in ("1_1","1_2","1_3","2_1","2_1p","2_2","2_2p","2_3p"):
    imgs[n]=cv2.imread(f"/workspace/images/{n}.jpg")
res={}
for n,img in imgs.items():
    if img is None: continue
    res[n]=process_image(img,p)["items"]
    print(n, len(res[n]))

def score(img, it):
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
    if not deltas: return 0.0
    dark_frac=sum(1 for d in deltas if d>1.0)/len(deltas)
    depth=float(np.mean(deltas))
    return max(0.0,depth)*dark_frac

# check pairs on 2_1 vs 2_1p
pairs=[]
for v in res["2_1"]:
    bd=1e9;best=None
    for v2 in res["2_1p"]:
        d=(v["center_x"]-v2["center_x"])**2+(v["center_y"]-v2["center_y"])**2
        if d<bd: bd,best=d,v2
    if bd<40**2: pairs.append((v,best))
SA=np.array([score(imgs["2_1"],a) for a,b in pairs])
SB=np.array([score(imgs["2_1p"],b) for a,b in pairs])
print("pairwise: A p50/p90:", np.percentile(SA,[50,90]).round(2), "B p10/p50:", np.percentile(SB,[10,50]).round(2))
for thr in (1,1.5,2,2.5,3,3.5,4):
    print(f"thr={thr}: FP={np.mean(SA>thr):.3f} TP={np.mean(SB>thr):.3f}")
# other images distributions
for n in ("1_1","1_2","1_3","2_2","2_2p","2_3p"):
    S=np.array([score(imgs[n],it) for it in res[n][:120]])
    print(n, "percentiles", np.percentile(S,[10,50,90]).round(2))
