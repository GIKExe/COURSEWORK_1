import cv2, numpy as np, pandas as pd
from scipy.signal import find_peaks

PROG = {'1_1':0.068465,'1_2':0.066822,'1_3':0.063253,'2_1':0.066270,
        '2_1p':0.064464,'2_2':0.063195,'2_2p':0.063944,'2_3p':0.070832}

def profile_for(name):
    img = cv2.imread(f'images/{name}.jpg')
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_y = int(H*0.7)
    roi = gray[roi_y:, :]
    _, bw = cv2.threshold(roi, 110, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 0.02*roi.size]
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    rot = ang + (90 if rw < rh else 0)
    M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H*sin + W*cos), int(H*cos + W*sin)
    M[0, 2] += (nW - W)/2; M[1, 2] += (nH - H)/2
    rg = cv2.warpAffine(gray, M, (nW, nH), borderValue=0)
    mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(mask, [c + np.array([[0, roi_y]])], -1, 255, -1)
    rm = cv2.warpAffine(mask, M, (nW, nH), borderValue=0)
    ys, xs = np.where(rm > 128)
    g = rg[ys.min():ys.max()+1, xs.min():xs.max()+1]
    m = rm[ys.min():ys.max()+1, xs.min():xs.max()+1] > 128
    h = g.shape[0]
    top = ((g[:h//2] < 110) & m[:h//2]).sum(axis=0)
    bot = ((g[h//2:] < 110) & m[h//2:]).sum(axis=0)
    prof = (top if top.std() > bot.std() else bot).astype(float)
    prof = np.convolve(prof, np.ones(3)/3, mode='same')
    return prof

def ac_peak(prof, lo, hi):
    x = prof - prof.mean()
    ac = np.correlate(x, x, mode='full')[len(x)-1:]
    ac = ac/ac[0]
    seg = ac[lo:hi+1]
    i = int(np.argmax(seg)) + lo
    return i, float(ac[i]), ac

rows = []
for name in PROG:
    prof = profile_for(name)
    lag1, a1, ac = ac_peak(prof, 12, 19)
    lag5, a5, _ = ac_peak(prof, 65, 92)
    lag10, a10, _ = ac_peak(prof, 135, 175)
    # prefer 5 mm harmonic; check 10 mm consistency
    per_mm_5 = lag5/5.0
    per_mm_10 = lag10/10.0
    per_mm_1 = lag1
    # robust: median of available periods within 4%
    cands = [per_mm_1, per_mm_5, per_mm_10]
    med = np.median(cands)
    good = [c for c in cands if abs(c-med) <= 0.04*med]
    per_mm = float(np.mean(good))
    mm_px = 1.0/per_mm
    rows.append(dict(photo=name, lag_1mm=lag1, ac_1mm=round(a1,3), lag_5mm=lag5, ac_5mm=round(a5,3),
                     lag_10mm=lag10, ac_10mm=round(a10,3), px_per_mm=round(per_mm,3),
                     mm_per_px_ruler=round(mm_px,6), mm_per_px_program=PROG[name],
                     rel_diff_pct=round((mm_px-PROG[name])/PROG[name]*100,2)))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(r'results\ruler_scale_check.csv', index=False)
print('mean rel diff %', df.rel_diff_pct.mean(), 'median', df.rel_diff_pct.median())
print('SD rel diff %', df.rel_diff_pct.std())
