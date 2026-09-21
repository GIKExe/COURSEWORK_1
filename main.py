import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


class BlurDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор размытых областей")
        self.root.geometry("1200x800")
        
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
        tk.Label(self.control_frame, text="Порог:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=50.0)
        self.threshold_scale = tk.Scale(
            self.control_frame, 
            from_=10, 
            to=200, 
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            resolution=5
        )
        self.threshold_scale.pack(side=tk.LEFT, padx=5)
        
        tk.Label(self.control_frame, text="Размер ядра:").pack(side=tk.LEFT, padx=5)
        self.kernel_size_var = tk.IntVar(value=15)
        self.kernel_scale = tk.Scale(
            self.control_frame,
            from_=5,
            to=51,
            orient=tk.HORIZONTAL,
            variable=self.kernel_size_var,
            resolution=2
        )
        self.kernel_scale.pack(side=tk.LEFT, padx=5)
        
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
        
        if canvas_width < 100 or canvas_height < 100:
            canvas_width = 800
            canvas_height = 600
        
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
        kernel_size = self.kernel_size_var.get()
        
        # Перевод в градации серого
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float64)
        
        # Применение фильтра Лапласиана
        laplacian = cv2.Laplacian(gray_float, cv2.CV_64F)
        
        # Вычисление локальной дисперсии с помощью скользящего окна
        # Используем размытие для получения среднего значения
        kernel = np.ones((kernel_size, kernel_size), np.float64) / (kernel_size * kernel_size)
        
        # Среднее значение квадрата Лапласиана
        laplacian_sq = laplacian ** 2
        mean_sq = cv2.filter2D(laplacian_sq, -1, kernel)
        
        # Квадрат среднего значения Лапласиана
        mean_lap = cv2.filter2D(laplacian, -1, kernel)
        mean_lap_sq = mean_lap ** 2
        
        # Дисперсия = E[X²] - (E[X])²
        variance_map = mean_sq - mean_lap_sq
        
        # Обеспечиваем неотрицательность
        variance_map = np.maximum(variance_map, 0)
        
        # Нормализация карты дисперсии
        variance_normalized = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX)
        variance_normalized = variance_normalized.astype(np.uint8)
        
        # Пороговая бинаризация (низкая дисперсия = размытие)
        _, binary_map = cv2.threshold(
            variance_normalized, 
            threshold, 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # Морфологические операции для очистки
        morph_kernel = np.ones((5, 5), np.uint8)
        
        # Закрытие для заполнения мелких отверстий
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
        
        # Открытие для удаления мелких шумов
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        
        # Размытие для создания плавных границ
        binary_map = cv2.GaussianBlur(binary_map, (21, 21), 0)
        
        # Создание маски размытых областей
        h, w = gray.shape
        blur_mask = (binary_map > 127).astype(np.uint8) * 255
        
        # Наложение полупрозрачного красного цвета на размытые области
        result = self.original_image.copy()
        
        # Создаем красный оверлей
        red_overlay = np.zeros_like(result)
        red_overlay[:, :] = [0, 0, 255]  # BGR
        
        # Применяем маску с прозрачностью
        mask_bool = blur_mask > 0
        alpha = 0.5
        
        for c in range(3):
            result[mask_bool, c] = (
                alpha * red_overlay[mask_bool, c] + 
                (1 - alpha) * result[mask_bool, c]
            ).astype(np.uint8)
        
        # Поиск и отрисовка контуров
        contours, _ = cv2.findContours(
            binary_map, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Отрисовка контуров зелёным цветом
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        self.processed_image = result
        self.display_image(result)
        
        # Вывод статистики
        blur_percentage = (np.sum(mask_bool) / (h * w)) * 100
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