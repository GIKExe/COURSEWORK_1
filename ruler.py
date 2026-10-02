#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Калибровка mm/px по фотографии линейки.
Подход: контурный анализ штрихов, касающихся края линейки.
"""

import cv2
import numpy as np
import sys
import os


class RulerCalibrator:
    def __init__(self, image_path: str):
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить: {image_path}")
        self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        self.H, self.W = self.img_gray.shape
        self._crop = None
        self._crop_mask = None
        self._preprocess()

    # ------------------------------------------------------------------
    #  Предобработка: найти линейку → выпрямить → обрезать
    # ------------------------------------------------------------------

    def _preprocess(self):
        gray = self.img_gray

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern, iterations=2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Линейка не найдена")

        ruler_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(ruler_cnt) < self.H * self.W * 0.005:
            raise ValueError("Объект слишком мал для линейки")

        rect = cv2.minAreaRect(ruler_cnt)
        (cx, cy), (rw, rh), angle = rect
        rot_angle = angle + 90 if rw < rh else angle

        M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        nW = int(self.H * sin_a + self.W * cos_a)
        nH = int(self.H * cos_a + self.W * sin_a)
        M[0, 2] += (nW - self.W) / 2
        M[1, 2] += (nH - self.H) / 2

        rotated = cv2.warpAffine(gray, M, (nW, nH), borderValue=0)

        mask_full = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.drawContours(mask_full, [ruler_cnt], -1, 255, -1)
        rotated_mask = cv2.warpAffine(mask_full, M, (nW, nH), borderValue=0)

        ys, xs = np.where(rotated_mask > 128)
        if len(xs) == 0:
            raise ValueError("Маска пуста после поворота")

        pad = 3
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(nW, xs.max() + pad + 1), min(nH, ys.max() + pad + 1)

        self._crop = rotated[y0:y1, x0:x1]
        self._crop_mask = rotated_mask[y0:y1, x0:x1]

        ch, cw = self._crop.shape
        if ch > cw:
            self._crop = cv2.rotate(self._crop, cv2.ROTATE_90_CLOCKWISE)
            self._crop_mask = cv2.rotate(self._crop_mask, cv2.ROTATE_90_CLOCKWISE)

    # ------------------------------------------------------------------
    #  Основной метод
    # ------------------------------------------------------------------

    def calibrate(self, debug: bool = False) -> float:
        crop = self._crop
        mask = self._crop_mask
        ch, cw = crop.shape

        # Адаптивный порог: тёмные штрихи → белые
        bs = max(11, (min(ch, cw) // 6) | 1)
        adaptive = cv2.adaptiveThreshold(
            crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=bs, C=8
        )

        # Слабое сужение маски (срезу кромку, но сохраняем длину штрихов)
        kern_e = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_e = cv2.erode(mask, kern_e, iterations=1)
        tick_bin = cv2.bitwise_and(adaptive, mask_e)

        # Вертикальное закрытие: сшивает прерывистые штрихи,
        # точки/мусор при этом не вырастают
        kern_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        tick_bin = cv2.morphologyEx(tick_bin, cv2.MORPH_CLOSE, kern_v)

        # Границы линейки для каждого столбца (учитывает остаточный наклон)
        mb = mask_e > 128
        col_any = mb.any(axis=0)
        top_edge = np.where(col_any, mb.argmax(axis=0), -1)
        bot_edge = np.where(col_any, ch - 1 - mb[::-1].argmax(axis=0), -1)

        contours, _ = cv2.findContours(tick_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Геометрические пороги
        min_area = max(5.0, ch * 0.05)     # тонкие штрихи проходят
        max_area = ch * cw * 0.005
        min_h = ch * 0.05
        max_h = ch * 0.75
        max_w = max(6, int(cw * 0.004))
        tol = max(3, int(ch * 0.04))       # допуск касания края

        ticks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if not (min_h <= bh <= max_h) or bw > max_w:
                continue
            if bh / max(bw, 1) < 2.5:
                continue

            # ГЛАВНЫЙ ФИЛЬТР: штрих обязан касаться края линейки.
            # Цифры и надписи лежат в середине → отбрасываются.
            cxi = min(max(bx + bw // 2, 0), cw - 1)
            te, be = top_edge[cxi], bot_edge[cxi]
            if te < 0:
                continue
            if by <= te + tol:
                edge = 'top'
            elif by + bh >= be - tol:
                edge = 'bottom'
            else:
                continue

            M = cv2.moments(cnt)
            cx = M['m10'] / M['m00'] if M['m00'] > 0 else bx + bw / 2.0
            cy = M['m01'] / M['m00'] if M['m00'] > 0 else by + bh / 2.0
            ticks.append(dict(cx=cx, cy=cy, bx=bx, by=by,
                              bw=bw, bh=bh, edge=edge))

        if len(ticks) < 8:
            raise ValueError(f"Мало штрихов найдено: {len(ticks)}")

        # Доминирующий край
        n_top = sum(1 for t in ticks if t['edge'] == 'top')
        use_top = n_top >= len(ticks) - n_top
        edge_ticks = [t for t in ticks if (t['edge'] == 'top') == use_top]
        if len(edge_ticks) < 8:
            edge_ticks = ticks

        # Сортировка + дедупликация (зазор меньше 0.5 мм шага)
        edge_ticks.sort(key=lambda t: t['cx'])
        min_gap = max(2, round(cw * 0.0006))
        uniq = [edge_ticks[0]]
        for t in edge_ticks[1:]:
            if t['cx'] - uniq[-1]['cx'] > min_gap:
                uniq.append(t)
            elif t['bh'] > uniq[-1]['bh']:
                uniq[-1] = t

        if len(uniq) < 8:
            raise ValueError(f"После дедупликации мало штрихов: {len(uniq)}")

        # Расстояния между соседними штрихами → мода = 1 мм
        x_pos = np.array([t['cx'] for t in uniq], dtype=np.float64)
        distances = np.diff(x_pos)
        d_1mm = self._dominant_distance(distances)
        mm_per_px = 1.0 / d_1mm

        if debug:
            self._save_debug(tick_bin, uniq, d_1mm, mm_per_px)

        return mm_per_px

    # ------------------------------------------------------------------
    #  Кластеризация расстояний
    # ------------------------------------------------------------------

    @staticmethod
    def _dominant_distance(distances: np.ndarray) -> float:
        d = np.asarray(distances, dtype=np.float64)

        q1, q3 = np.percentile(d, [20, 80])
        iqr = q3 - q1
        clean = d[(d >= q1 - 2 * iqr) & (d <= q3 + 2 * iqr)]
        if len(clean) < 3:
            clean = d

        n_bins = max(15, min(60, len(clean)))
        counts, edges = np.histogram(clean, bins=n_bins)
        peak_bin = int(np.argmax(counts))
        d_mode = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0

        tol = d_mode * 0.20
        cluster = clean[np.abs(clean - d_mode) <= tol]
        if len(cluster) < 2:
            return float(np.median(clean))
        return float(np.mean(cluster))

    # ------------------------------------------------------------------
    #  Отладка
    # ------------------------------------------------------------------

    def _save_debug(self, tick_bin, ticks, d_1mm, mm_per_px):
        vis = cv2.cvtColor(self._crop, cv2.COLOR_GRAY2BGR)
        for t in ticks:
            bx, by, bw, bh = t['bx'], t['by'], t['bw'], t['bh']
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
            cv2.circle(vis, (int(round(t['cx'])), int(round(t['cy']))),
                       2, (0, 0, 255), -1)

        txt = (f"mm/px = {mm_per_px:.6f}   1mm = {d_1mm:.2f}px   "
               f"ticks = {len(ticks)}")
        cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.imwrite('debug_ticks.png', vis)
        cv2.imwrite('debug_binary.png', tick_bin)


# ======================================================================

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'ruler_photo.jpg'
    if not os.path.exists(path):
        print(f"Файл не найден: {path}")
        sys.exit(1)

    cal = RulerCalibrator(path)
    mm_per_px = cal.calibrate(debug=True)

    print(f"mm/px  = {mm_per_px:.6f}")
    print(f"px/mm  = {1.0 / mm_per_px:.2f}")


if __name__ == '__main__':
    main()