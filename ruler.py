#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
  КАЛИБРОВКА mm/px ПО ФОТОГРАФИИ ЛИНЕЙКИ
==============================================================================
  Два подхода:
    1. Анализ 1D профиля интенсивности + детекция пиков (sub-pixel)
    2. Контурный анализ штрихов (геометрическая фильтрация)

  Вход: светлая линейка на чёрном коврике, штрихи 1 мм + 0.5 мм (начало)
  Зависимости: opencv-python, numpy, Pillow
==============================================================================
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Optional
import sys
import os


class RulerCalibrator:
    """
    Определение соотношения мм/пиксель по фотографии линейки
    на тёмном фоне.
    """

    def __init__(self, image_path: str):
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить: {image_path}")

        self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        self.H, self.W = self.img_gray.shape

        # Результаты предобработки (заполняются в _preprocess)
        self._crop = None          # выпрямленный кроп линейки (grayscale)
        self._crop_mask = None     # маска линейки в кропе
        self._ruler_angle = 0.0

        self._preprocess()

    # ==================================================================
    #  ПРЕДОБРАБОТКА: поиск линейки → выравнивание → обрезка
    # ==================================================================

    def _preprocess(self):
        """Находит линейку, выпрямляет и обрезает."""
        gray = self.img_gray.copy()

        # 1) Сглаживание + бинаризация (Отсу)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 2) Морфология: закрыть щели, убрать пылинки
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern, iterations=2)

        # 3) Находим крупнейший контур — линейка
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("Линейка не найдена (нет крупных светлых объектов)")

        ruler_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(ruler_cnt) < self.H * self.W * 0.005:
            raise ValueError("Найденный объект слишком мал для линейки")

        # 4) Ориентация через minAreaRect
        rect = cv2.minAreaRect(ruler_cnt)
        (cx, cy), (rw, rh), angle = rect
        self._ruler_angle = angle + 90 if rw < rh else angle

        # 5) Поворот изображения
        M = cv2.getRotationMatrix2D((cx, cy), self._ruler_angle, 1.0)
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        nW = int(self.H * sin_a + self.W * cos_a)
        nH = int(self.H * cos_a + self.W * sin_a)
        M[0, 2] += (nW - self.W) / 2
        M[1, 2] += (nH - self.H) / 2

        rotated = cv2.warpAffine(gray, M, (nW, nH), borderValue=0)

        # Маска линейки
        mask_full = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.drawContours(mask_full, [ruler_cnt], -1, 255, -1)
        rotated_mask = cv2.warpAffine(mask_full, M, (nW, nH), borderValue=0)

        # 6) Обрезка по bounding box маски
        ys, xs = np.where(rotated_mask > 128)
        if len(xs) == 0:
            raise ValueError("Маска линейки пуста после поворота")

        pad = 3
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(nW, xs.max() + pad + 1), min(nH, ys.max() + pad + 1)

        self._crop = rotated[y0:y1, x0:x1]
        self._crop_mask = rotated_mask[y0:y1, x0:x1]

        # Гарантируем горизонтальную ориентацию (W > H)
        ch, cw = self._crop.shape
        if ch > cw:
            self._crop = cv2.rotate(self._crop, cv2.ROTATE_90_CLOCKWISE)
            self._crop_mask = cv2.rotate(self._crop_mask, cv2.ROTATE_90_CLOCKWISE)

    # ==================================================================
    #  ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ==================================================================

    @staticmethod
    def _find_peaks(signal: np.ndarray, min_dist: int = 5,
                    min_prominence: float = None) -> np.ndarray:
        """
        Локальные максимумы в 1D сигнале с sub-pixel уточнением
        (параболическая интерполяция) и жадной фильтрацией по
        выраженности (prominence).
        """
        n = len(signal)
        if n < 3:
            return np.array([])

        if min_prominence is None:
            min_prominence = (signal.max() - signal.min()) * 0.12

        # Сырые локальные максимумы
        raw = []
        for i in range(1, n - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                raw.append(i)
        if not raw:
            return np.array([])

        # Sub-pixel уточнение
        refined = []
        for p in raw:
            if 1 <= p <= n - 2:
                a, b, c = signal[p - 1], signal[p], signal[p + 1]
                denom = 2.0 * (2 * b - a - c)
                if abs(denom) > 1e-12:
                    offset = max(-0.5, min(0.5, (a - c) / denom))
                    refined.append(p + offset)
                else:
                    refined.append(float(p))
            else:
                refined.append(float(p))
        refined = np.array(refined)

        # Prominence каждого пика
        proms = np.empty(len(refined))
        for i, r in enumerate(refined):
            idx = int(round(r))
            left  = signal[max(0, idx - min_dist * 3):idx]
            right = signal[idx + 1:min(n, idx + min_dist * 3 + 1)]
            l_min = left.min()  if len(left)  else signal[idx]
            r_min = right.min() if len(right) else signal[idx]
            proms[i] = signal[idx] - max(l_min, r_min)

        # Фильтр по prominence
        keep = proms >= min_prominence
        refined, proms = refined[keep], proms[keep]
        if len(refined) == 0:
            return np.array([])

        # Жадная фильтрация по min_dist (от самых выраженных)
        order = np.argsort(-proms)
        selected = []
        for idx in order:
            pos = refined[idx]
            if all(abs(pos - s) >= min_dist for s in selected):
                selected.append(pos)

        return np.sort(np.array(selected))

    @staticmethod
    def _cluster_distances(distances: np.ndarray) -> Dict:
        """
        Кластеризация расстояний между штрихами.
        
        Доминирующий кластер → 1 мм.
        Дополнительно ищет кластеры 0.5 мм и 5 мм для валидации.
        """
        if len(distances) < 3:
            raise ValueError(f"Мало расстояний: {len(distances)}")

        d = np.asarray(distances, dtype=np.float64)

        # Грубые выбросы (IQR × 2)
        q1, q3 = np.percentile(d, [20, 80])
        iqr = q3 - q1
        clean = d[(d >= q1 - 2 * iqr) & (d <= q3 + 2 * iqr)]
        if len(clean) < 3:
            clean = d

        # Модальное расстояние через гистограмму
        n_bins = max(15, min(50, len(clean)))
        counts, edges = np.histogram(clean, bins=n_bins)
        peak_bin = int(np.argmax(counts))
        d_mode = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0

        # Уточнение: среднее значений в ±20 % от моды
        tol = d_mode * 0.20
        cl_1mm = clean[np.abs(clean - d_mode) <= tol]
        if len(cl_1mm) < 2:
            d_1mm = float(np.median(clean))
            cl_1mm = clean[np.abs(clean - d_1mm) <= d_1mm * 0.25]
        else:
            d_1mm = float(np.mean(cl_1mm))

        std_1mm = float(np.std(cl_1mm)) if len(cl_1mm) > 1 else None

        # Кластер 0.5 мм
        d_half = d_1mm / 2.0
        cl_half = clean[np.abs(clean - d_half) <= d_1mm * 0.12]
        d_05 = float(np.mean(cl_half)) if len(cl_half) >= 2 else None

        # Кластер 5 мм
        d_five = d_1mm * 5.0
        cl_five = clean[np.abs(clean - d_five) <= d_1mm * 0.5]
        d_5 = float(np.mean(cl_five)) if len(cl_five) >= 1 else None

        return {
            'd_1mm_px': d_1mm, 'd_05mm_px': d_05, 'd_5mm_px': d_5,
            'std_1mm': std_1mm, 'n_1mm': len(cl_1mm),
            'n_05mm': len(cl_half), 'n_total': len(clean),
        }

    @staticmethod
    def _confidence(cl: Dict) -> float:
        """Оценка уверенности (0…1)."""
        s = 0.4
        s += min(0.25, cl['n_1mm'] * 0.01)
        if cl['std_1mm'] is not None and cl['d_1mm_px'] > 0:
            cv = cl['std_1mm'] / cl['d_1mm_px']
            s += 0.2 if cv < 0.03 else (0.1 if cv < 0.07 else (0.05 if cv < 0.12 else 0))
        if cl['d_05mm_px'] and cl['d_1mm_px'] > 0:
            ratio = cl['d_1mm_px'] / cl['d_05mm_px']
            if 1.85 < ratio < 2.15:
                s += 0.1
        return min(1.0, s)

    # ==================================================================
    #  ПОДХОД 1 — АНАЛИЗ 1D ПРОФИЛЯ ИНТЕНСИВНОСТИ
    # ==================================================================

    def approach1_profile(self, debug: bool = False) -> Dict:
        """
        Строим 1D профиль яркости по столбцам выпрямленного кропа.
        Тёмные штрихи → впадины → инверсия → пики.
        Доминирующее межпиковое расстояние = 1 мм.
        """
        crop, mask = self._crop, self._crop_mask
        ch, cw = crop.shape

        # Перебираем перекрывающиеся горизонтальные полосы,
        # выбираем полосу с максимальной дисперсией (там штрихи)
        n_strips = 5
        strip_h = ch // n_strips
        best_prof, best_score = None, -1.0

        for s in range(n_strips):
            y0 = s * strip_h
            y1 = min(ch, y0 + strip_h * 2)
            region = crop[y0:y1, :]
            rmask  = mask[y0:y1, :]

            prof = np.full(cw, 255.0, dtype=np.float64)
            for x in range(cw):
                vals = region[:, x][rmask[:, x] > 128]
                if len(vals) > 2:
                    prof[x] = np.percentile(vals, 10)

            valid = prof[prof < 255]
            if len(valid) < 10:
                continue
            score = float(np.std(valid))
            if score > best_score:
                best_score = score
                best_prof = prof

        if best_prof is None:
            raise ValueError("Не удалось построить профиль")

        prof = best_prof

        # Нормализация → инверсия
        valid = prof < 255
        pmin, pmax = prof[valid].min(), prof[valid].max()
        if pmax - pmin < 5:
            raise ValueError("Профиль слишком плоский")

        norm = np.where(valid, (prof - pmin) / (pmax - pmin), 0.0)
        inverted = 1.0 - norm                        # штрихи → пики

        ks = max(3, cw // 300) | 1                   # нечётный kernel
        smooth = cv2.GaussianBlur(
            inverted.reshape(1, -1).astype(np.float32), (ks, 1), 0
        ).flatten().astype(np.float64)

        # Детекция пиков
        min_d = max(4, int(cw * 0.003))
        peaks = self._find_peaks(smooth, min_dist=min_d)
        if len(peaks) < 4:
            raise ValueError(f"Мало пиков: {len(peaks)}")

        # Кластеризация
        distances = np.diff(peaks)
        cl = self._cluster_distances(distances)

        result = {
            'method': 'Profile Peak Detection',
            'mm_per_px': 1.0 / cl['d_1mm_px'],
            'px_per_mm': cl['d_1mm_px'],
            'n_peaks': len(peaks),
            'clusters': cl,
            'confidence': self._confidence(cl),
            'peaks': peaks,
            'profile': prof,
        }
        if debug:
            self._debug_profile(result, smooth)
        return result

    # ==================================================================
    #  ПОДХОД 2 — КОНТУРНЫЙ АНАЛИЗ ШТРИХОВ
    # ==================================================================

    def approach2_contours(self, debug: bool = False) -> Dict:
        """
        Адаптивная бинаризация внутри маски → контуры тёмных штрихов
        → геометрическая фильтрация → сортировка → расстояния.
        """
        crop, mask = self._crop, self._crop_mask
        ch, cw = crop.shape

        # Адаптивный порог (инверсный — штрихи становятся белыми)
        bs = max(11, (min(ch, cw) // 6) | 1)
        adaptive = cv2.adaptiveThreshold(
            crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=bs, C=8
        )

        # Сужаем маску (убираем краевые артефакты)
        kern_e = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        inner = cv2.erode(mask, kern_e, iterations=2)
        tick_bin = cv2.bitwise_and(adaptive, inner)

        # Морфо-чистка
        tick_bin = cv2.morphologyEx(
            tick_bin, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        )

        # Контуры
        contours, _ = cv2.findContours(
            tick_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Геометрическая фильтрация
        min_area = max(4, ch * 0.5)
        max_area = ch * cw * 0.005
        min_h    = ch * 0.04
        max_w    = max(10, cw * 0.015)

        ticks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh < min_h or bw > max_w:
                continue
            if bh / max(bw, 1) < 2.0:          # aspect ratio
                continue

            M = cv2.moments(cnt)
            cx = M['m10'] / M['m00'] if M['m00'] > 0 else bx + bw / 2
            cy = M['m01'] / M['m00'] if M['m00'] > 0 else by + bh / 2

            edge = 'top' if by <= (ch - by - bh) else 'bottom'
            ticks.append(dict(cx=cx, cy=cy, bx=bx, by=by,
                              bw=bw, bh=bh, area=area, edge=edge))

        if len(ticks) < 4:
            raise ValueError(f"Мало штрихов: {len(ticks)}")

        # Доминирующий край
        n_top = sum(1 for t in ticks if t['edge'] == 'top')
        edge_ticks = [t for t in ticks if (t['edge'] == 'top') == (n_top >= len(ticks) - n_top)]
        if len(edge_ticks) < 4:
            edge_ticks = ticks

        # Сортировка по X + дедупликация
        edge_ticks.sort(key=lambda t: t['cx'])
        min_gap = max(3, cw * 0.002)
        uniq = [edge_ticks[0]]
        for t in edge_ticks[1:]:
            if t['cx'] - uniq[-1]['cx'] > min_gap:
                uniq.append(t)
            elif t['bh'] > uniq[-1]['bh']:
                uniq[-1] = t

        if len(uniq) < 4:
            raise ValueError(f"После дедупликации: {len(uniq)}")

        x_pos = np.array([t['cx'] for t in uniq], dtype=np.float64)
        distances = np.diff(x_pos)
        cl = self._cluster_distances(distances)

        result = {
            'method': 'Contour Tick Detection',
            'mm_per_px': 1.0 / cl['d_1mm_px'],
            'px_per_mm': cl['d_1mm_px'],
            'n_ticks_raw': len(ticks),
            'n_ticks_used': len(uniq),
            'clusters': cl,
            'confidence': self._confidence(cl),
            'tick_x': x_pos,
        }
        if debug:
            self._debug_contours(result, tick_bin, uniq)
        return result

    # ==================================================================
    #  ИТОГОВЫЙ РЕЗУЛЬТАТ
    # ==================================================================

    def calibrate(self, debug: bool = True) -> Dict:
        """Запуск обоих подходов + взвешенное объединение."""
        out = {}
        for key, fn in [('approach1', self.approach1_profile),
                        ('approach2', self.approach2_contours)]:
            try:
                out[key] = fn(debug=debug)
            except Exception as e:
                out[key] = {'error': str(e)}

        valid = [v for v in out.values() if 'error' not in v]
        if not valid:
            errs = '\n'.join(f"  {k}: {v.get('error','?')}" for k, v in out.items())
            raise RuntimeError("Оба подхода завершились ошибкой:\n" + errs)

        if len(valid) == 2:
            c1, c2 = valid[0]['confidence'], valid[1]['confidence']
            w1, w2 = c1 / (c1 + c2), c2 / (c1 + c2)
            mm_px = valid[0]['mm_per_px'] * w1 + valid[1]['mm_per_px'] * w2
            agr = (abs(valid[0]['mm_per_px'] - valid[1]['mm_per_px'])
                   / ((valid[0]['mm_per_px'] + valid[1]['mm_per_px']) / 2) * 100)
        else:
            mm_px = valid[0]['mm_per_px']
            agr = None

        out['final'] = {
            'mm_per_px': mm_px,
            'px_per_mm': 1.0 / mm_px,
            'agreement_pct': agr,
        }
        return out

    # ==================================================================
    #  ОТЛАДОЧНАЯ ВИЗУАЛИЗАЦИЯ (OpenCV + Pillow)
    # ==================================================================

    def _debug_profile(self, result: Dict, smooth: np.ndarray):
        """Сохраняет отладочные файлы для подхода 1."""
        # 1) Кроп с отмеченными пиками (OpenCV)
        vis = cv2.cvtColor(self._crop, cv2.COLOR_GRAY2BGR)
        for px in result['peaks']:
            x = int(round(px))
            cv2.line(vis, (x, 0), (x, vis.shape[0] - 1), (0, 0, 255), 1)
        cv2.imwrite('debug_a1_peaks.png', vis)

        # 2) График профиля (Pillow)
        n = len(smooth)
        pw = max(n, 800)
        ph = 300
        img = Image.new('RGB', (pw, ph), 'white')
        draw = ImageDraw.Draw(img)

        sx = pw / n
        smin, smax = smooth.min(), smooth.max()
        rng = smax - smin if smax > smin else 1.0

        pts = []
        for i in range(n):
            x = int(i * sx)
            y = int(ph - 20 - (smooth[i] - smin) / rng * (ph - 40))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill='black', width=1)

        for px in result['peaks']:
            idx = int(round(px))
            if 0 <= idx < n:
                x = int(idx * sx)
                y = int(ph - 20 - (smooth[idx] - smin) / rng * (ph - 40))
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill='red')
                draw.line([(x, y + 4), (x, ph - 5)], fill='red', width=1)

        img.save('debug_a1_profile.png')

    def _debug_contours(self, result: Dict, tick_bin: np.ndarray,
                        ticks: list):
        """Сохраняет отладочные файлы для подхода 2."""
        vis = cv2.cvtColor(self._crop, cv2.COLOR_GRAY2BGR)
        for t in ticks:
            bx, by, bw, bh = t['bx'], t['by'], t['bw'], t['bh']
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
            cv2.circle(vis, (int(round(t['cx'])), int(round(t['cy']))),
                       2, (0, 0, 255), -1)
        cv2.imwrite('debug_a2_ticks.png', vis)
        cv2.imwrite('debug_a2_binary.png', tick_bin)


# ======================================================================
#  ТОЧКА ВХОДА
# ======================================================================

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'ruler_photo.jpg'
    if not os.path.exists(path):
        print(f"Файл не найден: {path}")
        print("Использование: python ruler_calibration.py <путь_к_фото>")
        sys.exit(1)

    print(f"Загрузка: {path}")
    cal = RulerCalibrator(path)
    print("Калибровка...")
    res = cal.calibrate(debug=True)

    W = 62
    print(f"\n{'=' * W}")
    print("  РЕЗУЛЬТАТЫ КАЛИБРОВКИ  mm / px".center(W))
    print(f"{'=' * W}")

    labels = {
        'approach1': 'Подход 1: 1D-профиль (детекция пиков)',
        'approach2': 'Подход 2: Контурный анализ штрихов',
    }
    for key in ('approach1', 'approach2'):
        print(f"\n  {labels[key]}")
        print("  " + "-" * (W - 4))
        r = res[key]
        if 'error' in r:
            print(f"  ✗ Ошибка: {r['error']}")
        else:
            c = r['clusters']
            print(f"  mm/px       = {r['mm_per_px']:.6f}")
            print(f"  px/mm       = {r['px_per_mm']:.2f}")
            print(f"  Уверенность = {r['confidence']:.0%}")
            std_s = f"{c['std_1mm']:.2f}" if c['std_1mm'] else "—"
            print(f"  1 мм  = {c['d_1mm_px']:.2f} px  (σ={std_s}, n={c['n_1mm']})")
            if c['d_05mm_px']:
                print(f"  0.5 мм= {c['d_05mm_px']:.2f} px  (n={c['n_05mm']})")

    f = res['final']
    print(f"\n{'=' * W}")
    print("  ИТОГО".center(W))
    print(f"{'=' * W}")
    print(f"  mm/px  = {f['mm_per_px']:.6f}")
    print(f"  px/mm  = {f['px_per_mm']:.2f}")
    if f['agreement_pct'] is not None:
        status = "✓" if f['agreement_pct'] < 2.0 else "⚠"
        print(f"  {status} Расхождение подходов: {f['agreement_pct']:.2f}%")
    print(f"{'=' * W}\n")


if __name__ == '__main__':
    main()