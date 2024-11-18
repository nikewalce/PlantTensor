'''
функция разбивает исходные данные на тренировочный и валидационный списки по 
которым будут формироваться две независимых части данных для всех алгоритмов
'''

import glob, sys, time
import numpy as np
from tqdm import tqdm

source_folder = './New Plant Diseases Dataset(Augmented)'
train_file = source_folder+'/train.txt'
val_file = source_folder+'/val.txt'

# получение списка файлов из заданных папок
def get_all_files(source_folders, patterns):
    if not isinstance(source_folders, list):
        source_folders = [source_folders]
    if not isinstance(patterns, list):
        patterns = [patterns]
    files = []
    for source_folder in source_folders:
        for pattern in patterns:
            files = files + glob.glob(source_folder+'/**/'+pattern, recursive = True)
    files = np.unique(np.array([file.replace('\\','/') for file in files]))
    return files

train_img_files = get_all_files(source_folders = source_folder+'/train', patterns = ['*.jpeg', '*.png','*.jpg', '*.webp'])
train_classes = [img_file.split('/')[-2] for img_file in train_img_files]

val_img_files = get_all_files(source_folders = source_folder+'/valid', patterns = ['*.jpeg', '*.png','*.jpg', '*.webp'])
val_classes = [img_file.split('/')[-2] for img_file in val_img_files]

for val_class in val_classes:
    if val_class not in train_classes:
        sys.exit("Класс {} отсутствует в тренировочных данных!".format(val_class))

# Тренировочная и валидационная таблицы
train_txt = []
val_txt = []
# Главный цикл
n = len(train_img_files)
pbar = tqdm(total = n)
for i in range(n):
    train_txt.append([train_img_files[i], train_classes[i]])
    pbar.update(1)
pbar.close()
time.sleep(0.3)

n = len(val_img_files)
pbar = tqdm(total = n)
for i in range(n):
    val_txt.append([val_img_files[i], val_classes[i]])
    pbar.update(1)
pbar.close()

train_txt = np.array(train_txt)
val_txt = np.array(val_txt)
# Создание выходных файлов таблиц
np.savetxt(fname = train_file, X = np.array(train_txt, dtype = str), fmt = "%s", delimiter = '\t')
np.savetxt(fname = val_file, X = np.array(val_txt, dtype = str), fmt = "%s", delimiter = '\t')
# сохранить таблицу классов
u_classes = np.sort(np.unique(train_txt[:,1]))
np.savetxt(source_folder+'/u_classes.txt', u_classes, fmt = '%s', delimiter = '\t', encoding = 'utf8')
print("Число классов: {}".format(len(u_classes)))
