import os
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2

from tkinter import filedialog

class ImageSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Выбор изображения")
        
        # Переменная для хранения пути к папке с изображениями
        self.image_folder = None
        
        # Переменная для хранения выбранного изображения
        self.selected_image = None
        
        # Создание кнопки для выбора папки
        self.folder_button = tk.Button(root, text="Выбрать папку", command=self.choose_folder)
        self.folder_button.pack(pady=10)
        
        # Создание Canvas для отображения изображений
        self.canvas = tk.Canvas(root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Добавление Scrollbar
        self.scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Фрейм для изображений
        self.image_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor=tk.NW)
        
        # Обработка изменения размера окна
        self.image_frame.bind("<Configure>", self.on_frame_configure)
        
        # Списки для хранения изображений и миниатюр
        self.images = []
        self.thumbnails = []

    def choose_folder(self):
        """Открывает диалог выбора папки и загружает изображения."""
        self.image_folder = filedialog.askdirectory()  # Диалог выбора папки
        if self.image_folder:
            self.load_images()

    def load_images(self):
        """Загружает изображения из выбранной папки и отображает их как миниатюры."""
        # Очистка предыдущих изображений
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.images.clear()
        self.thumbnails.clear()
        
        # Загрузка изображений из папки
        self.image_files = sorted(
            [f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda x: x.lower()  # Сортировка в алфавитном порядке (без учета регистра)
        )
        
        if not self.image_files:
            return
        
        # Загрузка и отображение миниатюр
        for idx, image_file in enumerate(self.image_files):
            image_path = os.path.join(self.image_folder, image_file)
            try:
                image = Image.open(image_path)
                image.thumbnail((100, 100))  # Создание миниатюры
                photo = ImageTk.PhotoImage(image)
                
                # Сохранение ссылки на изображение
                self.images.append(photo)
                
                # Создание фрейма для миниатюры и названия файла
                frame = tk.Frame(self.image_frame)
                frame.grid(row=idx // 10, column=idx % 10, padx=5, pady=5)  # 4 колонки
                
                # Кнопка с миниатюрой
                button = tk.Button(frame, image=photo, command=lambda f=image_file: self.select_image(f))
                button.pack()
                
                # Название файла
                label = tk.Label(frame, text=image_file)
                label.pack()
            except Exception as e:
                print(f"Ошибка загрузки изображения {image_file}: {e}")

    def select_image(self, image_file):
        """Обрабатывает выбор изображения и закрывает окно."""
        self.selected_image = image_file
        self.root.destroy()  # Закрытие окна

    def on_frame_configure(self, event):
        """Обновляет область прокрутки при изменении размера окна."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

def show_image(image, title):
    image_pil = Image.fromarray(image).convert("RGB")
    image_pil.thumbnail((500, 500))
    image_tk = ImageTk.PhotoImage(image_pil)
    mask_label = tk.Label(root, image=image_tk, text=title, compound="top")
    mask_label.image = image_tk
    mask_label.pack(side="left", padx=10, pady=10)

def circularity(cnt):
  area = cv2.contourArea(cnt)
  perimeter = cv2.arcLength(cnt, True)
  return 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

def process_card(card, image, image1, blur=True):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  card = cv2.morphologyEx(cv2.drawContours(np.zeros_like(image), [card], -1, 255, -1), cv2.MORPH_CLOSE, kernel, iterations=10)

  corr_cnt = cv2.findContours(card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
  x, text_y = int(np.mean(corr_cnt[:,:,0]).item()), int(np.mean(corr_cnt[:,:,1]).item())
  epsilon = 0.01 * cv2.arcLength(corr_cnt, True)
  number = len(cv2.approxPolyDP(corr_cnt, epsilon, True)) // 4
  write = f"{number}"
  (text_width, text_height), baseline = cv2.getTextSize(write, cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.7, 2)
  cv2.rectangle(image1, (x, text_y), (x + text_width, text_y - text_height), (255, 255, 255), -1)
  cv2.putText(image1, write, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
              0.6, (255,0,0), 2)
  cv2.drawContours(image1, [corr_cnt], -1, (0,255,0), 3)
  
  cur = image & card
  cur = cv2.Canny(cur, 50, 100, edges=True, L2gradient=True)
  cur_cnts, _ = cv2.findContours(cur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  answer = []
  for i in range(len(cur_cnts)):
    r = cv2.morphologyEx(cv2.drawContours(np.zeros_like(image), [cur_cnts[i]], -1, 255, -1), cv2.MORPH_CLOSE, kernel, iterations=10)
    r_cnts, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(r_cnts[0])
    epsilon = 0.01 * cv2.arcLength(r_cnts[0], True)
    approx = cv2.approxPolyDP(r_cnts[0], epsilon, True)
    circ = circularity(r_cnts[0])
    if area >= 2000 and circ >= 0.2 and area <= 10000:
      if circ >= 0.82:
        answer.append((0, r_cnts[0], cv2.isContourConvex(approx), circ))
      else:
        answer.append((len(approx), r_cnts[0], cv2.isContourConvex(approx), circ))
  return answer, number

def process_image(path, count_label):
  image = cv2.imread(path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  show_image(image, "Исходное изображение")
  image1 = image.copy()
  image0 = image.copy()
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
  image = clahe.apply(image[:,:,2])
  mask = cv2.inRange(image, 0, 150)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  for _ in range(1):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
  cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 3000]
  mask = cv2.drawContours(np.zeros_like(image), cnts, -1, 255, -1)
  cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  answers = []
  number = 0
  for cnt in cnts:
    card_res = process_card(cnt, image, image1)
    answers += card_res[0]
    number += card_res[1]
  show_image(image1, "Карточки")
  lens = [answer[0] for answer in answers]
  circs = [answer[3] for answer in answers]
  convs = [answer[2] for answer in answers]
  answers = [answer[1] for answer in answers]
  res = cv2.drawContours(image0, answers, -1, (0,255,0), 3)
  for i, l in enumerate(lens):
    x,y,w,h = cv2.boundingRect(answers[i])
    text_y = y - 10 if y - 10 > 10 else y + h + 30
    if l == 0:
      write = "S"
    else:
      write = f"P{l}"
    if convs[i]:
      write += "C"
    (text_width, text_height), baseline = cv2.getTextSize(write, cv2.FONT_HERSHEY_SIMPLEX,
                                                          0.7, 2)
    cv2.rectangle(res, (x, text_y), (x + text_width, text_y - text_height), (255, 255, 255), -1)
    cv2.putText(res, write, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,0,0), 2)
  show_image(res, "Фигуры")
  count_label.config(text=f"Количество карточек: {number}")
  return lens, answers, convs

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Детектор карточек и фигур")
    app = ImageSelectorApp(root)
    root.mainloop()
    
    if app.selected_image:
        path = f"{app.image_folder}/{app.selected_image}"
        root = tk.Tk()
        root.title("Детектор карточек и фигур")

        count_label = tk.Label(root, text="", font=("Arial", 16))
        count_label.pack()

        process_image(path, count_label)

        root.mainloop()
    else:
        print("Изображение не выбрано.")