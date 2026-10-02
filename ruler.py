#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Калибровка mm/px по фотографии линейки.
Подход: контурный анализ штрихов линейки.
"""

import cv2
import numpy as np
import sys
import os


class RulerCalibrator:
    """
    Определение mm/px по фотографии линейки на тёмном фоне.
    Светлая линейка, штрихи 1 мм (и 0.5 мм в начале).
    """

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

        # Бинаризация (линейка светлее фона)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Морфология: закрыть щели, убрать пылинки
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern, iterations=2)

        # Крупнейший контур = линейка
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Линейка не найдена")

        ruler_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(ruler_cnt) < self.H * self.W * 0.005:
            raise ValueError("Объект слишком мал для линейки")

        # Ориентация
        rect = cv2.minAreaRect(ruler_cnt)
        (cx, cy), (rw, rh), angle = rect
        rot_angle = angle + 90 if rw < rh else angle

        # Поворот
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

        # Обрезка
        ys, xs = np.where(rotated_mask > 128)
        if len(xs) == 0:
            raise ValueError("Маска пуста после поворота")

        pad = 3
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(nW, xs.max() + pad + 1), min(nH, ys.max() + pad + 1)

        self._crop = rotated[y0:y1, x0:x1]
        self._crop_mask = rotated_mask[y0:y1, x0:x1]

        # Горизонтальная ориентация
        ch, cw = self._crop.shape
        if ch > cw:
            self._crop = cv2.rotate(self._crop, cv2.ROTATE_90_CLOCKWISE)
            self._crop_mask = cv2.rotate(self._crop_mask, cv2.ROTATE_90_CLOCKWISE)

    # ------------------------------------------------------------------
    #  Основной метод: контурный анализ штрихов
    # ------------------------------------------------------------------

    def calibrate(self, debug: bool = False) -> float:
        """
        Возвращает mm_per_px.
        """
        crop = self._crop
        mask = self._crop_mask
        ch, cw = crop.shape

        # Адаптивный порог: тёмные штрихи → белый
        bs = max(11, (min(ch, cw) // 6) | 1)
        adaptive = cv2.adaptiveThreshold(
            crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=bs, C=8
        )

        # Сужаем маску (краевые артефакты)
        kern_e = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        inner = cv2.erode(mask, kern_e, iterations=2)
        tick_bin = cv2.bitwise_and(adaptive, inner)

        # Убираем мелкий шум
        tick_bin = cv2.morphologyEx(
            tick_bin, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        )

        # Контуры
        contours, _ = cv2.findContours(tick_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Фильтрация по геометрии
        min_area = max(4, ch * 0.5)
        max_area = ch * cw * 0.005
        min_h = ch * 0.04
        max_w = max(10, cw * 0.015)

        ticks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh < min_h or bw > max_w:
                continue
            if bh / max(bw, 1) < 2.0:
                continue

            M = cv2.moments(cnt)
            cx = M['m10'] / M['m00'] if M['m00'] > 0 else bx + bw / 2
            cy = M['m01'] / M['m00'] if M['m00'] > 0 else by + bh / 2
            edge = 'top' if by <= (ch - by - bh) else 'bottom'

            ticks.append(dict(cx=cx, cy=cy, bx=bx, by=by,
                              bw=bw, bh=bh, edge=edge))

        if len(ticks) < 4:
            raise ValueError(f"Мало штрихов найдено: {len(ticks)}")

        # Доминирующий край
        n_top = sum(1 for t in ticks if t['edge'] == 'top')
        use_top = n_top >= len(ticks) - n_top
        edge_ticks = [t for t in ticks if (t['edge'] == 'top') == use_top]
        if len(edge_ticks) < 4:
            edge_ticks = ticks

        # Сортировка + дедупликация
        edge_ticks.sort(key=lambda t: t['cx'])
        min_gap = max(3, cw * 0.002)
        uniq = [edge_ticks[0]]
        for t in edge_ticks[1:]:
            if t['cx'] - uniq[-1]['cx'] > min_gap:
                uniq.append(t)
            elif t['bh'] > uniq[-1]['bh']:
                uniq[-1] = t

        if len(uniq) < 4:
            raise ValueError(f"После дедупликации мало штрихов: {len(uniq)}")

        # Расстояния между соседними штрихами
        x_pos = np.array([t['cx'] for t in uniq], dtype=np.float64)
        distances = np.diff(x_pos)

        # Кластеризация: доминирующее расстояние = 1 мм
        d_1mm = self._dominant_distance(distances)
        mm_per_px = 1.0 / d_1mm

        if debug:
            self._save_debug(tick_bin, uniq, x_pos, d_1mm, mm_per_px)

        return mm_per_px

    # ------------------------------------------------------------------
    #  Кластеризация расстояний
    # ------------------------------------------------------------------

    @staticmethod
    def _dominant_distance(distances: np.ndarray) -> float:
        """Находит доминирующее расстояние (= 1 мм) в пикселях."""
        d = np.asarray(distances, dtype=np.float64)

        # Выбросы
        q1, q3 = np.percentile(d, [20, 80])
        iqr = q3 - q1
        clean = d[(d >= q1 - 2 * iqr) & (d <= q3 + 2 * iqr)]
        if len(clean) < 3:
            clean = d

        # Мода через гистограмму
        n_bins = max(15, min(50, len(clean)))
        counts, edges = np.histogram(clean, bins=n_bins)
        peak_bin = int(np.argmax(counts))
        d_mode = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0

        # Уточнение: среднее в ±20% от моды
        tol = d_mode * 0.20
        cluster = clean[np.abs(clean - d_mode) <= tol]
        if len(cluster) < 2:
            return float(np.median(clean))
        return float(np.mean(cluster))

    # ------------------------------------------------------------------
    #  Отладка
    # ------------------------------------------------------------------

    def _save_debug(self, tick_bin, ticks, x_pos, d_1mm, mm_per_px):
        vis = cv2.cvtColor(self._crop, cv2.COLOR_GRAY2BGR)
        for t in ticks:
            bx, by, bw, bh = t['bx'], t['by'], t['bw'], t['bh']
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
            cv2.circle(vis, (int(round(t['cx'])), int(round(t['cy']))),
                       2, (0, 0, 255), -1)

        # Текст результата
        txt = f"mm/px = {mm_per_px:.6f}   1mm = {d_1mm:.2f}px   ticks={len(ticks)}"
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