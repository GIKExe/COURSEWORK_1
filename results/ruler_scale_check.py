import cv2, numpy as np
from scipy.signal import find_peaks

def measure(name):
    img = cv2.imread(f'images/{name}.jpg')
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_y = int(H*0.7)
    roi = gray[roi_y:, :]
    _, bw = cv2.threshold(roi, 110, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 0.02*roi.size]
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    rot = ang + (90 if rw < rh else 0)
    M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW, nH = int(H*sin + W*cos), int(H*cos + W*sin)
    M[0,2] += (nW - W)/2; M[1,2] += (nH - H)/2
    rg = cv2.warpAffine(gray, M, (nW, nH), borderValue=0)
    mask = np.zeros((H,W), np.uint8)
    cv2.drawContours(mask, [c + np.array([[0, roi_y]])], -1, 255, -1)
    rm = cv2.warpAffine(mask, M, (nW, nH), borderValue=0)
    ys, xs = np.where(rm > 128)
    g = rg[ys.min():ys.max()+1, xs.min():xs.max()+1]
    m = rm[ys.min():ys.max()+1, xs.min():xs.max()+1] > 128
    h = g.shape[0]
    top = ((g[:h//2] < 110) & m[:h//2]).sum(axis=0)
    bot = ((g[h//2:] < 110) & m[h//2:]).sum(axis=0)
    prof = top if top.std() > bot.std() else bot
    prof = np.convolve(prof, np.ones(3)/3, mode='same')
    peaks, _ = find_peaks(prof, distance=8, prominence=3)
    heights = prof[peaks]
    thr = 0.72*heights.max()
    tall = np.sort(peaks[heights >= thr])
    # merge peaks closer than 40 px
    merged=[]
    for p in tall:
        if merged and p-merged[-1] < 60:
            merged[-1] = (merged[-1]+p)//2
        else:
            merged.append(p)
    tall=np.array(merged)
    d=np.diff(tall)
    d2=d[(d>90)&(d<260)]
    scale_px_per_mm = np.median(d2)/10 if len(d2) else np.nan
    return dict(name=name, npeaks=len(peaks), maxh=float(heights.max()), thr=float(thr),
                tall=tall, diffs=d2, px_per_mm=float(scale_px_per_mm),
                mm_per_px=float(10/scale_px_per_mm) if scale_px_per_mm else np.nan)

prog = {'1_1':0.068465,'1_2':0.066822,'1_3':0.063253,'2_1':0.066270,'2_1p':0.064464,'2_2':0.063195,'2_2p':0.063944,'2_3p':0.070832}
for name in ['1_1','1_2','1_3','2_1','2_1p','2_2','2_2p','2_3p']:
    r=measure(name)
    print(name, 'npeaks',r['npeaks'],'maxh',round(r['maxh'],1),'tall n',len(r['tall']),
          'px/mm',round(r['px_per_mm'],3) if r['px_per_mm']==r['px_per_mm'] else None,
          'mm/px',round(r['mm_per_px'],6) if r['mm_per_px']==r['mm_per_px'] else None,
          'program',prog[name],'ratio',round(r['mm_per_px']/prog[name],4) if r['mm_per_px']==r['mm_per_px'] else None)
    print('   tall', r['tall'][:30], 'diffs', r['diffs'][:20])
