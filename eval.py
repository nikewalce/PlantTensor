# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
tf.keras.backend.clear_session()

# Загрузка необходимых библиотек
import argparse
import pickle as pkl
import numpy as np

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

import cv2

from en_lite import EfficientNetLiteB0


print("Запуск с параметрами:")  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # размер изображения
    parser.add_argument('--image_size', type = int, default = 224)
    # файл для сохранения весов лучшей модели
    parser.add_argument('--best_weights', type = str, default='./best_weights.h5')
    # информация о классах
    parser.add_argument('--u_classes_file', type = str, default='./u_classes.txt')
    # парсинг параметров  
    opt = parser.parse_args()
print(opt)

# Загрузка информации по классам
u_classes = np.loadtxt(opt.u_classes_file, dtype = str, delimiter = '\t')

# Инициализация нейронной сети
m = EfficientNetLiteB0(input_shape = (opt.image_size, opt.image_size, 3), 
                       return_logits = True,
                       n_classes = len(u_classes))

# Загрузим последние веса модели, если они были ранее получены
if opt.best_weights is not None:
    if os.path.exists(opt.best_weights):
        m.load_weights(opt.best_weights, by_name = True, skip_mismatch = True)
        print("Веса модели загружены!")
    else:
        print("Веса модели не найдены!")

# получение прогнозов
@tf.function
def get_prediction(img):
    img = tf.cast(img, dtype = tf.float32)
    img = tf.expand_dims(img, axis = 0)
    logits = m(img, training = False)[0]
    return logits

# загрузка изображения
def load_img(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    # корректировка формата
    if len(img.shape) == 3 and img.shape[-1] == 3: 
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) == 2: 
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if len(img.shape) == 3 and img.shape[-1] == 4: 
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

def load_image_check():
    global m, opt, labels
    # Получить путь к файлу
    filetypes = (('png или jpeg файлы', '*.png *.jpg'),)
    filename = fd.askopenfilename(
        title = "Анализировать изображение...",
        initialdir = os.getcwd().replace('\\','/'),
        filetypes = filetypes)
    # Загрузка изображения
    img = load_img(filename)
    # пересжатие в рабочее разрешение
    img = cv2.resize(img, (opt.image_size, opt.image_size))
    # пропускаем через модель
    pr_class_logits = get_prediction(img).numpy()
    pr_class = u_classes[np.argmax(pr_class_logits)]
    # определение классов
    lbl_class_plant['text'] = 'Класс растения: ' + str(pr_class.split("___")[0])
    lbl_class_disease['text'] = 'Класс болезни: ' + str(pr_class.split("___")[1])
    print("Проанализирован файл: {}".format(filename))

root = tk.Tk()
root.title("Анализ болезней растений")
root.geometry("500x150")

# стандартная кнопка
btn = ttk.Button(text = "Анализировать изображение...", command = load_image_check)
btn.pack(pady = 20)

exit_btn = ttk.Button(text = "Выход из программы", command = root.destroy)
exit_btn.pack()

# результаты
lbl_class_plant = ttk.Label(text = 'Класс растения: None',
                            width = 100, 
                            anchor="w")
lbl_class_plant.pack(pady = 5, padx = 15)
lbl_class_disease = ttk.Label(text = 'Класс болезни: None',
                              width = 100, 
                              anchor="w")
lbl_class_disease.pack(pady = 5, padx = 15)

root.mainloop()