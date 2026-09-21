import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


class BlurDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор размытых областей (Defocus Blur)")
        self.root.geometry("1400x900")
        
        # Панель управления
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.btn_load = tk.Button(
            self.control_frame, 
            text="Выбрать изображение", 
            command=self.load_image,
            font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5
        )
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        tk.Label(self.control_frame, text="Порог:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=0.3)
        self.threshold_scale = tk.Scale(
            self.control_frame, from_=0.05, to=0.8, orient=tk.HORIZONTAL,
            variable=self.threshold_var, resolution=0.01
        )
        self.threshold_scale.pack(side=tk.LEFT, padx=5)
        
        tk.Label(self.control_frame, text="Размер ядра:").pack(side=tk.LEFT, padx=5)
        self.kernel_size_var = tk.IntVar(value=31)
        self.kernel_scale = tk.Scale(
            self.control_frame, from_=11, to=71, orient=tk.HORIZONTAL,
            variable=self.kernel_size_var, resolution=2
        )
        self.kernel_scale.pack(side=tk.LEFT, padx=5)
        
        tk.Label(self.control_frame, text="Мин. контраст:").pack(side=tk.LEFT, padx=5)
        self.min_contrast_var = tk.DoubleVar(value=15.0)
        self.min_contrast_scale = tk.Scale(
            self.control_frame, from_=5, to=50, orient=tk.HORIZONTAL,
            variable=self.min_contrast_var, resolution=1
        )
        self.min_contrast_scale.pack(side=tk.LEFT, padx=5)
        
        self.btn_process = tk.Button(
            self.control_frame, text="Обработать",
            command=self.process_image,
            font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5,
            state=tk.DISABLED
        )
        self.btn_process.pack(side=tk.LEFT, padx=5)
        
        self.btn_show_maps = tk.Button(
            self.control_frame, text="Карты (отладка)",
            command=self.show_debug_maps,
            font=("Arial", 12), bg="#FF9800", fg="white", padx=10, pady=5,
            state=tk.DISABLED
        )
        self.btn_show_maps.pack(side=tk.LEFT, padx=5)
        
        self.image_frame = tk.Frame(root, bg="gray")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.image_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.original_image = None
        self.processed_image = None
        self.debug_maps = {}
        self.image_path = None
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("Все файлы", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
                return
            self.btn_process.config(state=tk.NORMAL)
            self.btn_show_maps.config(state=tk.DISABLED)
            self.display_image(self.original_image)
            
    def display_image(self, image):
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 100 or canvas_height < 100:
            canvas_width, canvas_height = 800, 600
        
        h, w = image_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_resized = cv2.resize(image_rgb, (new_w, new_h))
        
        pil_image = Image.fromarray(image_resized)
        self.photo = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.photo)
        
    def compute_blur_detection(self, gray, kernel_size, min_contrast):
        """
        Детекция defocus blur.
        
        Ключевая идея:
        - В РЁЗКОЙ области: много высокочастотных деталей, высокий локальный контраст
        - В РАЗМЫТОЙ области (defocus): структура есть, но高频 детали подавлены
        - В ОДНОРОДНОЙ области: структуры нет вообще, контраст ~0
        
        Алгоритм:
        1. Вычисляем "потерю деталей" при размытии (detail_loss)
           - В резких областях: большое значение (много деталей теряется)
           - В размытых областях: маленькое значение (деталей уже мало)
           - В однородных: тоже маленькое
        
        2. Вычисляем локальный контраст (contrast)
           - В резких и размытых областях со структурой: ненулевой
           - В однородных: ~0
        
        3. Нормализуем: blur_score = 1 - (detail_loss / contrast)
           - Резкая область: detail_loss >> contrast → blur_score → 0
           - Размытая область: detail_loss < contrast → blur_score → высокое
           - Однородная: contrast ~0 → игнорируем через порог min_contrast
        """
        gray_float = gray.astype(np.float64)
        
        # 1. Локальный контраст (std в гауссовом окне)
        mean_img = cv2.GaussianBlur(gray_float, (kernel_size, kernel_size), 0)
        mean_img_sq = cv2.GaussianBlur(gray_float**2, (kernel_size, kernel_size), 0)
        variance = np.maximum(mean_img_sq - mean_img**2, 0)
        contrast = np.sqrt(variance)
        
        # 2. Потеря деталей при размытии
        # Размытие с меньшим ядром - "эталон" локальной структуры
        small_blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
        detail = np.abs(gray_float - small_blur)
        local_detail = cv2.GaussianBlur(detail, (kernel_size, kernel_size), 0)
        
        # 3. Нормализованная мера размытия
        # В резких областях: local_detail высокий относительно contrast
        # В размытых областях: local_detail низкий относительно contrast
        contrast_safe = np.maximum(contrast, 1.0)
        sharpness_ratio = local_detail / contrast_safe
        
        # blur_score: 0 = резко, 1 = размыто
        # Нормализуем sharpness_ratio через процентили
        ratio_min = np.percentile(sharpness_ratio[contrast > min_contrast], 10) if np.sum(contrast > min_contrast) > 0 else 0
        ratio_max = np.percentile(sharpness_ratio[contrast > min_contrast], 90) if np.sum(contrast > min_contrast) > 0 else 1
        
        if ratio_max > ratio_min:
            normalized_ratio = (sharpness_ratio - ratio_min) / (ratio_max - ratio_min)
        else:
            normalized_ratio = np.zeros_like(sharpness_ratio)
        
        normalized_ratio = np.clip(normalized_ratio, 0, 1)
        blur_score = 1.0 - normalized_ratio  # инвертируем: высокое = размыто
        
        # Маска однородных областей (игнорируем)
        uniform_mask = contrast < min_contrast
        
        # В однородных областях blur_score = 0 (не размытие)
        blur_score[uniform_mask] = 0
        
        return blur_score, contrast, local_detail, uniform_mask
        
    def process_image(self):
        if self.original_image is None:
            return
            
        threshold = self.threshold_var.get()
        kernel_size = self.kernel_size_var.get()
        min_contrast = self.min_contrast_var.get()
        
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Вычисление карты размытия
        blur_score, contrast, local_detail, uniform_mask = self.compute_blur_detection(
            gray, kernel_size, min_contrast
        )
        
        # Сохраняем для отладки
        self.debug_maps = {
            'blur_score': blur_score,
            'contrast': contrast,
            'local_detail': local_detail,
            'uniform_mask': uniform_mask
        }
        self.btn_show_maps.config(state=tk.NORMAL)
        
        # Бинаризация
        binary_map = (blur_score > threshold).astype(np.uint8) * 255
        
        # Морфологические операции
        morph_kernel = np.ones((7, 7), np.uint8)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        
        # Плавные границы
        binary_map = cv2.GaussianBlur(binary_map, (31, 31), 0)
        
        h, w = gray.shape
        blur_mask = (binary_map > 127).astype(np.uint8) * 255
        
        # Наложение красного цвета
        result = self.original_image.copy()
        red_overlay = np.zeros_like(result)
        red_overlay[:, :] = [0, 0, 255]
        
        mask_bool = blur_mask > 0
        alpha = 0.5
        for c in range(3):
            result[mask_bool, c] = (alpha * red_overlay[mask_bool, c] + (1 - alpha) * result[mask_bool, c]).astype(np.uint8)
        
        # Контуры
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        cv2.drawContours(result, large_contours, -1, (0, 255, 0), 2)
        
        self.processed_image = result
        self.display_image(result)
        
        blur_percentage = (np.sum(mask_bool) / (h * w)) * 100
        messagebox.showinfo("Результат", f"Найдено {len(large_contours)} размытых областей\nПроцент размытия: {blur_percentage:.1f}%")
    
    def show_debug_maps(self):
        if not self.debug_maps:
            return
        
        blur_score = self.debug_maps['blur_score']
        contrast = self.debug_maps['contrast']
        uniform_mask = self.debug_maps['uniform_mask']
        
        # Визуализация blur_score
        blur_vis = (blur_score * 255).astype(np.uint8)
        blur_colored = cv2.applyColorMap(blur_vis, cv2.COLORMAP_JET)
        
        # Визуализация контраста
        contrast_norm = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        contrast_colored = cv2.applyColorMap(contrast_norm, cv2.COLORMAP_VIRIDIS)
        
        # Комбинированная визуализация
        h, w = blur_score.shape
        combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
        combined[:, :w] = blur_colored
        combined[:, w:2*w] = contrast_colored
        combined[:, 2*w:] = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        self.display_image(combined)


def main():
    root = tk.Tk()
    app = BlurDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()