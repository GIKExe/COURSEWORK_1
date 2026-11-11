import cv2, numpy as np, os
from scipy.signal import find_peaks

def ruler_profile(name, verbose=False):
    img = cv2.imread(f'images/{name}.jpg')
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ruler = bright object in bottom part
    roi = gray[int(H*0.7):, :]
    _, bw = cv2.threshold(roi, 110, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 0.02*roi.size]
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), ang = rect
    # make long side horizontal
    rot = ang
    if rw < rh:
        rot = ang + 90
    M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW, nH = int(H*sin + W*cos), int(H*cos + W*sin)
    M[0,2] += (nW - W)/2; M[1,2] += (nH - H)/2
    rot_gray = cv2.warpAffine(gray, M, (nW, nH), borderValue=0)
    mask = np.zeros((H,W), np.uint8)
    cv2.drawContours(mask, [c + np.array([[0, int(H*0.7)]])], -1, 255, -1)
    rot_mask = cv2.warpAffine(mask, M, (nW, nH), borderValue=0)
    ys, xs = np.where(rot_mask > 128)
    y0,y1,x0,x1 = ys.min(), ys.max(), xs.min(), xs.max()
    g = rot_gray[y0:y1+1, x0:x1+1]
    m = rot_mask[y0:y1+1, x0:x1+1] > 128
    # count dark pixels per column inside mask
    dark = ((g < 110) & m).sum(axis=0)
    # find the edge: ticks dense where profile has high variance along columns in top vs bottom half
    h = g.shape[0]
    top = dark_top = ((g[:h//2] < 110) & m[:h//2]).sum(axis=0)
    bot = ((g[h//2:] < 110) & m[h//2:]).sum(axis=0)
    # ticks are where many dark pixels in the half adjacent to the scaled edge
    use_top = top.std() > bot.std()
    prof = top if use_top else bot
    # smooth
    k = np.ones(3)/3
    profs = np.convolve(prof, k, mode='same')
    peaks, props = find_peaks(profs, distance=8, prominence=3)
    heights = profs[peaks]
    if verbose:
        print(name, 'ruler crop', g.shape, 'edge', 'top' if use_top else 'bottom', 'npeaks', len(peaks))
        print('peak heights p50/p90/p99/max', np.percentile(heights,50), np.percentile(heights,90), np.percentile(heights,99), heights.max())
    return profs, peaks, heights, g

for name in ['2_1','1_2']:
    profs, peaks, heights, g = ruler_profile(name, verbose=True)
    order = np.argsort(heights)[::-1]
    tall = peaks[order[:40]]
    tall = np.sort(tall)
    print('tall peak positions sample:', tall[:40])
    d = np.diff(tall)
    print('tall spacings:', d)
