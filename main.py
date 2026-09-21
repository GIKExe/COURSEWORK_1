import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


class BlurDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор размытых областей")
        self.root.geometry("900x700")
        
        # Панель управления
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Кнопка выбора изображения
        self.btn_load = tk.Button(
            self.control_frame, 
            text="Выбрать изображение", 
            command=self.load_image,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5
        )
        self.btn_load.pack(side=tk.LEFT, padx=5)
        
        # Настройки параметров
        tk.Label(self.control_frame, text="Порог дисперсии:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=100.0)
        self.threshold_scale = tk.Scale(
            self.control_frame, 
            from_=10, 
            to=500, 
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            resolution=10
        )
        self.threshold_scale.pack(side=tk.LEFT, padx=5)
        
        tk.Label(self.control_frame, text="Размер блока:").pack(side=tk.LEFT, padx=5)
        self.block_size_var = tk.IntVar(value=20)
        self.block_scale = tk.Scale(
            self.control_frame,
            from_=5,
            to=50,
            orient=tk.HORIZONTAL,
            variable=self.block_size_var,
            resolution=1
        )
        self.block_scale.pack(side=tk.LEFT, padx=5)
        
        # Кнопка обработки
        self.btn_process = tk.Button(
            self.control_frame,
            text="Обработать",
            command=self.process_image,
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            padx=10,
            pady=5,
            state=tk.DISABLED
        )
        self.btn_process.pack(side=tk.LEFT, padx=5)
        
        # Область отображения изображения
        self.image_frame = tk.Frame(root, bg="gray")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.image_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Переменные
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        
    def load_image(self):
        """Загрузка изображения из файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Все файлы", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is None:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
                return
            
            self.btn_process.config(state=tk.NORMAL)
            self.display_image(self.original_image)
            
    def display_image(self, image):
        """Отображение изображения на canvas"""
        # Конвертация из BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Получение размеров canvas
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Масштабирование изображения под canvas
        h, w = image_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image_resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # Конвертация в формат PIL
        pil_image = Image.fromarray(image_resized)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Очистка canvas и отображение
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, 
            canvas_height // 2, 
            anchor=tk.CENTER,
            image=self.photo
        )
        
    def process_image(self):
        """Обработка изображения для обнаружения размытых областей"""
        if self.original_image is None:
            return
            
        # Параметры
        threshold = self.threshold_var.get()
        block_size = self.block_size_var.get()
        
        # Перевод в градации серого
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Применение фильтра Лапласиана
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Вычисление дисперсии в локальных блоках
        h, w = gray.shape
        blur_map = np.zeros((h, w), dtype=np.float64)
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = laplacian[y:y+block_size, x:x+block_size]
                variance = np.var(block)
                blur_map[y:y+block_size, x:x+block_size] = variance
        
        # Нормализация карты размытости
        blur_map_normalized = cv2.normalize(blur_map, None, 0, 255, cv2.NORM_MINMAX)
        blur_map_normalized = blur_map_normalized.astype(np.uint8)
        
        # Пороговая бинаризация (инвертированная: низкая дисперсия = размытие)
        _, binary_map = cv2.threshold(
            blur_map_normalized, 
            threshold, 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # Морфологические операции для очистки
        kernel = np.ones((5, 5), np.uint8)
        
        # Закрытие для заполнения мелких отверстий
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Открытие для удаления мелких шумов
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Размытие границ для более плавного перехода
        binary_map = cv2.GaussianBlur(binary_map, (15, 15), 0)
        
        # Поиск контуров размытых областей
        contours, _ = cv2.findContours(
            binary_map, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Создание маски размытых областей
        blur_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(blur_mask, contours, -1, 255, -1)
        
        # Наложение полупрозрачного красного цвета на размытые области
        result = self.original_image.copy()
        overlay = result.copy()
        
        # Создание красного слоя
        overlay[blur_mask > 127] = [0, 0, 255]  # BGR формат
        
        # Альфа-смешивание с прозрачностью 50%
        alpha = 0.5
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        
        # Отрисовка контуров для наглядности
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        self.processed_image = result
        self.display_image(result)
        
        # Вывод статистики
        blur_percentage = (np.sum(blur_mask > 127) / (h * w)) * 100
        messagebox.showinfo(
            "Результат обработки",
            f"Найдено {len(contours)} размытых областей\n"
            f"Процент размытия: {blur_percentage:.1f}%"
        )


def main():
    root = tk.Tk()
    app = BlurDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()