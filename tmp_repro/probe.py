import sys; sys.path.insert(0,"/workspace/src")
import numpy as np, cv2
from core.segmentation import process_image
from core.params import get_default_params

p = get_default_params()
img1 = cv2.imread("/workspace/images/2_1.jpg"); img2 = cv2.imread("/workspace/images/2_1p.jpg")
r1 = process_image(img1, p); r2 = process_image(img2, p)
c1 = {("x", round(x["center_x"]), round(x["center_y"])): x for x in r1["items"]}
c2 = list(r2["items"])
pairs=[]
for _,v in c1.items():
    bd=1e9; best=None
    for v2 in c2:
        d=(v["center_x"]-v2["center_x"])**2+(v["center_y"]-v2["center_y"])**2
        if d<bd: bd,best=d,v2
    if bd<40**2: pairs.append((v,best))
print("pairs:", len(pairs))

def stats(img, it):
    cnt = it["contour"].reshape(-1,2).astype(np.float64)
    ang = np.radians(it["angle_deg"]); ux,uy=np.cos(ang),np.sin(ang); vx,vy=-uy,ux
    x,y,w,h = it["bbox"]; pad=4
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
    sc=float(np.median(sv)); span=float(np.percentile(sv,97)-np.percentile(sv,3))
    core=np.abs(sv-sc)<=max(3.0,span*0.12)
    flank=np.abs(sv-sc)>max(3.0,span*0.30)
    depth=float(gv[flank].mean()-gv[core].mean())
    # dark-fraction along 4 segments
    bins=np.quantile(tv,[0,.25,.5,.75,1])
    seg_dark=0; nseg=0
    for k in range(4):
        m=core&(tv>=bins[k])&(tv<=bins[k+1])
        f=flank&(tv>=bins[k])&(tv<=bins[k+1])
        if m.sum()<5 or f.sum()<5: continue
        nseg+=1
        if gv[f].mean()-gv[m].mean()>1.0: seg_dark+=1
    # curvature across axis (groove -> U shape profile at mid-length)
    # gradient energy inside
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag=np.sqrt(gx*gx+gy*gy)
    grad=float(mag[sel].mean())
    return dict(depth=depth, dark_frac=seg_dark/max(1,nseg), grad=grad, core_mean=float(gv[core].mean()), flank_mean=float(gv[flank].mean()))

A=[];B=[]
for a,b in pairs:
    A.append(stats(img1,a)); B.append(stats(img2,b))
def summ(name,d):
    print(name, "depth %.2f darkfrac %.2f grad %.2f" % (np.mean([x['depth'] for x in d]), np.mean([x['dark_frac'] for x in d]), np.mean([x['grad'] for x in d])))
summ("no-groove(2_1)",A); summ("groove(2_1p)",B)
# distribution of combined score
for thr_d in (0.5,1.0,1.5,2.0):
    for thr_g in (2.0,3.0,4.0):
        predA=[ (x['depth']>thr_d and x['dark_frac']>=0.5) for x in A]
        predB=[ (x['depth']>thr_d and x['dark_frac']>=0.5) for x in B]
        print(f"d>{thr_d}: FP={np.mean(predA):.2f} TP={np.mean(predB):.2f}")
