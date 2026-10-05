"""
Калибровка мм/пиксель по фотографии линейки.

Модуль переделан для работы с изображением из памяти.
Основная функция:

	calibrate_ruler(image, debug=False) -> float

Параметры:
	image: numpy.ndarray
		Изображение в памяти. Поддерживается BGR, BGRA или grayscale.
	debug: bool
		Если True, сохранит отладочные изображения:
		- debug_ruler_ticks.png
		- debug_ruler_binary.png

Возврат:
	float
		Масштаб в мм/пиксель.
"""

import cv2
import numpy as np


__all__ = ["calibrate_ruler"]


def calibrate_ruler(image, debug=False):
	"""
	Возвращает мм/пиксель по фотографии линейки, переданной как ndarray.
	"""
	gray = _to_gray_uint8(image)

	crop, mask = _preprocess_ruler(gray)
	tick_bin, ticks, heights = _detect_ticks(crop, mask)

	x_pos = np.array([t["cx"] for t in ticks], dtype=np.float64)
	distances = np.diff(x_pos)

	# Шаг 1: грубый масштаб по сантиметровым штрихам.
	# Самые длинные штрихи = см, стоят через 10 мм.
	px_per_mm = _cm_scale(x_pos, heights)

	# Шаг 2: уточнение по всем штрихам (гипотезы 1 мм / 0.5 мм).
	if px_per_mm is not None:
		refined = _refine_scale(distances, px_per_mm)
		if refined is not None:
			px_per_mm = refined * 2
	else:
		px_per_mm = _dominant_distance(distances)

	if px_per_mm is None or not np.isfinite(px_per_mm) or px_per_mm <= 0:
		raise ValueError("Не удалось определить масштаб линейки")

	mm_per_px = 1.0 / float(px_per_mm)

	if debug:
		_save_ruler_debug(
			crop,
			tick_bin,
			ticks,
			heights,
			mm_per_px,
			px_per_mm
		)

	return float(mm_per_px)


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------

def _to_gray_uint8(image):
	if image is None:
		raise ValueError("Изображение не передано")

	if image.ndim == 3:
		channels = image.shape[2]

		if channels == 3:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		elif channels == 4:
			gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
		elif channels == 1:
			gray = image[:, :, 0].copy()
		else:
			gray = image[:, :, 0].copy()
	else:
		gray = image.copy()

	if gray.dtype != np.uint8:
		gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

	return gray


def _preprocess_ruler(gray):
	"""
	Предобработка:
	- найти линейку;
	- выпрямить;
	- обрезать;
	- ориентировать горизонтально.
	"""
	H, W = gray.shape

	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	_, binary = cv2.threshold(
		blurred,
		0,
		255,
		cv2.THRESH_BINARY + cv2.THRESH_OTSU
	)

	kern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

	binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern, iterations=3)
	binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern, iterations=2)

	contours, _ = cv2.findContours(
		binary,
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE
	)

	if not contours:
		raise ValueError("Линейка не найдена")

	ruler_cnt = max(contours, key=cv2.contourArea)

	# Если бинаризация выбрала почти весь кадр как объект,
	# пробуем инвертированную маску.
	full_area = float(H * W)

	if cv2.contourArea(ruler_cnt) > 0.95 * full_area:
		inverted = cv2.bitwise_not(binary)
		inverted = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kern, iterations=3)
		inverted = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kern, iterations=2)

		inv_contours, _ = cv2.findContours(
			inverted,
			cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE
		)

		if inv_contours:
			inv_cnt = max(inv_contours, key=cv2.contourArea)

			if cv2.contourArea(inv_cnt) < cv2.contourArea(ruler_cnt):
				ruler_cnt = inv_cnt

	if cv2.contourArea(ruler_cnt) < H * W * 0.005:
		raise ValueError("Объект слишком мал для линейки")

	rect = cv2.minAreaRect(ruler_cnt)
	(cx, cy), (rw, rh), angle = rect

	rot_angle = angle + 90 if rw < rh else angle

	M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)

	cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])

	nW = int(H * sin_a + W * cos_a)
	nH = int(H * cos_a + W * sin_a)

	M[0, 2] += (nW - W) / 2
	M[1, 2] += (nH - H) / 2

	rotated = cv2.warpAffine(
		gray,
		M,
		(nW, nH),
		borderValue=0
	)

	mask_full = np.zeros((H, W), dtype=np.uint8)
	cv2.drawContours(mask_full, [ruler_cnt], -1, 255, -1)

	rotated_mask = cv2.warpAffine(
		mask_full,
		M,
		(nW, nH),
		borderValue=0
	)

	ys, xs = np.where(rotated_mask > 128)

	if len(xs) == 0:
		raise ValueError("Маска пуста после поворота")

	pad = 3

	x0 = max(0, int(xs.min()) - pad)
	y0 = max(0, int(ys.min()) - pad)

	x1 = min(nW, int(xs.max()) + pad + 1)
	y1 = min(nH, int(ys.max()) + pad + 1)

	if x1 <= x0 or y1 <= y0:
		raise ValueError("Не удалось вырезать область линейки")

	crop = rotated[y0:y1, x0:x1]
	mask = rotated_mask[y0:y1, x0:x1]

	if crop.size == 0 or mask.size == 0:
		raise ValueError("Обрезанное изображение линейки пусто")

	ch, cw = crop.shape

	# Делаем линейку горизонтальной.
	if ch > cw:
		crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
		mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

	return crop, mask


def _detect_ticks(crop, mask):
	"""
	Детектирует штрихи линейки и возвращает:
	- бинарную маску штрихов;
	- список отсортированных уникальных штрихов;
	- массив высот штрихов.
	"""
	ch, cw = crop.shape

	# Адаптивный порог: тёмные штрихи -> белые.
	bs = max(11, (min(ch, cw) // 6) | 1)

	adaptive = cv2.adaptiveThreshold(
		crop,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV,
		blockSize=bs,
		C=8
	)

	kern_e = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	mask_e = cv2.erode(mask, kern_e, iterations=1)

	if not np.any(mask_e > 128):
		raise ValueError("Маска линейки пуста после эрозии")

	tick_bin = cv2.bitwise_and(adaptive, mask_e)

	# Вертикальное закрытие: сшивает прерывистые штрихи.
	kern_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
	tick_bin = cv2.morphologyEx(tick_bin, cv2.MORPH_CLOSE, kern_v)

	# Границы линейки поколоночно (учитывает остаточный наклон).
	mb = mask_e > 128
	col_any = mb.any(axis=0)

	top_edge = np.where(col_any, mb.argmax(axis=0), -1)
	bot_edge = np.where(col_any, ch - 1 - mb[::-1].argmax(axis=0), -1)

	contours, _ = cv2.findContours(
		tick_bin,
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE
	)

	min_area = max(5.0, ch * 0.05)
	max_area = ch * cw * 0.005

	min_h = ch * 0.05
	max_h = ch * 0.75

	max_w = max(6, int(cw * 0.004))
	tol = max(3, int(ch * 0.04))

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

		# Штрих обязан касаться края линейки (цифры/надписи — нет).
		cxi = min(max(bx + bw // 2, 0), cw - 1)

		te, be = top_edge[cxi], bot_edge[cxi]

		if te < 0:
			continue

		if by <= te + tol:
			edge = "top"
		elif by + bh >= be - tol:
			edge = "bottom"
		else:
			continue

		M = cv2.moments(cnt)

		cx = M["m10"] / M["m00"] if M["m00"] > 0 else bx + bw / 2.0
		cy = M["m01"] / M["m00"] if M["m00"] > 0 else by + bh / 2.0

		ticks.append(
			dict(
				cx=cx,
				cy=cy,
				bx=bx,
				by=by,
				bw=bw,
				bh=bh,
				edge=edge
			)
		)

	if len(ticks) < 8:
		raise ValueError(f"Мало штрихов найдено: {len(ticks)}")

	# Доминирующий край.
	n_top = sum(1 for t in ticks if t["edge"] == "top")
	use_top = n_top >= len(ticks) - n_top

	edge_ticks = [t for t in ticks if (t["edge"] == "top") == use_top]

	if len(edge_ticks) < 8:
		edge_ticks = ticks

	edge_ticks.sort(key=lambda t: t["cx"])

	min_gap = max(2, int(round(cw * 0.0006)))

	uniq = [edge_ticks[0]]

	for t in edge_ticks[1:]:
		if t["cx"] - uniq[-1]["cx"] > min_gap:
			uniq.append(t)
		elif t["bh"] > uniq[-1]["bh"]:
			uniq[-1] = t

	if len(uniq) < 8:
		raise ValueError(f"После дедупликации мало штрихов: {len(uniq)}")

	heights = np.array([t["bh"] for t in uniq], dtype=np.float64)

	return tick_bin, uniq, heights


def _cm_scale(x_pos, heights):
	"""
	Грубый px_per_mm по самым длинным (сантиметровым) штрихам.
	"""
	if len(x_pos) < 10:
		return None

	thr = np.percentile(heights, 88)
	xs = x_pos[heights >= thr]

	if len(xs) < 5:
		return None

	d = np.diff(xs)

	if len(d) == 0:
		return None

	med = float(np.median(d))

	if med <= 0:
		return None

	sel = d[np.abs(d - med) <= 0.15 * med]

	if len(sel) < 4:
		sel = d[np.abs(d - med) <= 0.30 * med]

	if len(sel) < 4:
		return None

	return float(np.mean(sel)) / 10.0


def _refine_scale(distances, px_per_mm):
	"""
	Уточнение: выбираем гипотезу решётки (1 мм или 0.5 мм),
	которая объясняет больше соседних расстояний, и усредняем её.

	Возвращает уже готовый px_per_mm, дополнительно умножать его не нужно.
	"""
	if distances is None or len(distances) == 0:
		return None

	if px_per_mm is None or px_per_mm <= 0:
		return None

	best_n, best_val = 0, None

	for k in (1.0, 0.5):
		target = px_per_mm * k

		sel = distances[np.abs(distances - target) <= 0.10 * target]

		if len(sel) >= 15 and len(sel) > best_n:
			best_n = len(sel)
			best_val = float(np.mean(sel)) / k

	return best_val


def _dominant_distance(distances):
	"""
	Fallback: мода соседних расстояний (если см-штрихи не найдены).
	"""
	d = np.asarray(distances, dtype=np.float64)
	d = d[d > 0]

	if len(d) == 0:
		return None

	med = float(np.median(d))

	if med <= 0:
		return None

	keep = d[(d > 0.4 * med) & (d < 1.6 * med)]

	if len(keep) < 3:
		keep = d

	counts, edges = np.histogram(keep, bins=40)

	peak = int(np.argmax(counts))
	d_mode = (edges[peak] + edges[peak + 1]) / 2.0

	sel = keep[np.abs(keep - d_mode) <= 0.10 * d_mode]

	if len(sel) < 2:
		return float(med)

	return float(np.mean(sel))


def _save_ruler_debug(crop, tick_bin, ticks, heights, mm_per_px, px_per_mm):
	"""
	Сохраняет отладочные изображения.
	"""
	try:
		thr = np.percentile(heights, 88)

		vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

		n_cm = 0

		for t, h in zip(ticks, heights):
			bx, by, bw, bh = t["bx"], t["by"], t["bw"], t["bh"]

			if h >= thr:
				# см-штрихи — синим
				cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
				n_cm += 1
			else:
				# остальные — зелёным
				cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)

			cv2.circle(
				vis,
				(int(round(t["cx"])), int(round(t["cy"]))),
				2,
				(0, 0, 255),
				-1
			)

		txt = (
			f"mm/px = {mm_per_px:.6f}   px/mm = {px_per_mm:.2f}   "
			f"ticks = {len(ticks)}   cm = {n_cm}"
		)

		cv2.putText(
			vis,
			txt,
			(10, 25),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 255, 255),
			2
		)

		cv2.imwrite("debug_ruler_ticks.png", vis)
		cv2.imwrite("debug_ruler_binary.png", tick_bin)

	except Exception:
		# Отладка не должна ломать основную калибровку.
		pass