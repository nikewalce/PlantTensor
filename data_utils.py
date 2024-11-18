'''
Данный модуль включает в себя базовые функции по препроцессингу исходных данных:
    - загрузка изображения;
    - аугментация;
    - расчет откликов.

Реализация преимущественно на skimage, numpy - как наиболее быстрые варианты
'''

from imgaug import augmenters as iaa # for augmentation of images
import numpy as np # for vector and matrixes
import matplotlib.pyplot as plt # for plots
from matplotlib.ticker import MaxNLocator
import cv2, sys, glob, os
#from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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

# аугментация температуры цветов (иначе не работает)
aug_wbt = iaa.ChangeColorTemperature(kelvin = (1000, 40000))
def aug_wbt_batch(images, random_state, parents, hooks):
    images = np.array(images,dtype=np.uint8)
    images = [aug_wbt(image = x) for x in images]
    return images

# Аугментатор изображений
aug_img = iaa.Sequential([
    # кроп случайной области
    iaa.Sometimes(0.25, [iaa.Crop(percent = (0.0, 0.15))]),
    # замораживание
    iaa.Sometimes(0.25, [iaa.imgcorruptlike.Frost(severity = (1, 3))]),
    # изменение положения, размера, угла поворота
    iaa.Sometimes(0.5,[
        iaa.Affine(
            scale = (0.75, 1.2),
            translate_percent = {"x": (-0.20, 0.20), "y": (-0.20, 0.20), "keep_size": False},
            shear = {"x": (-16, 16), "y": (-16, 16), "keep_size": False},
            rotate = (0, 360),
            fit_output = False, 
            mode= ['constant','edge'])
        ]),
    # изменение цвета, освещенности и т.п.
    iaa.Sometimes(0.25, [
        iaa.SomeOf((1, 2),[
            # аугментация температуры цветов
            iaa.Lambda(func_images=aug_wbt_batch),
            iaa.Add((100, -100), per_channel = 0.5), # добавление яркости
            iaa.Multiply((0.5, 1.5), per_channel = 0.5), # случайное изменение контрастности
            iaa.LinearContrast((0.5, 4.0), per_channel = 0.5), # случайное изменение контрастности
            iaa.GammaContrast((0.50, 2.50), per_channel = 0.5), # случайное изменение параметра gamma
            iaa.LogContrast(gain = (0.50, 1.50), per_channel = 0.5), # случайное изменение лог-контрастности
            iaa.MultiplyHue((-2.0, 2.0)), # очень чувствительный параметр hue
            iaa.MultiplySaturation((0.5, 2.5)), # изменение сатурации
            iaa.AddToHue((-50, 50)), # чувствительный параметр hue
            iaa.AddToSaturation((-50, 50)), # чувствительный параметр сатурации 
            # нормализация освещения
            iaa.AllChannelsCLAHE(clip_limit=(0.5, 5)),
            # улучшение цветопередачи
            iaa.pillike.EnhanceColor(factor = (0.5, 2.5)),
            # изменение диапазона цветов квантизацией
            iaa.UniformColorQuantizationToNBits(nb_bits=(3, 8))     # различные варианты изменения контрастности
            ])
        ]),
    # ухудшение качества путем добавление шумов, дропов и сглаживания
    iaa.Sometimes(0.25,[
        iaa.SomeOf((1),[
            # Туман
            iaa.BlendAlpha((0.25,0.5),iaa.Clouds()),            
            # Сдвиг каналов
            iaa.BlendAlpha((0.25, 0.5), 
                            iaa.Sequential([
                                iaa.TranslateX(px=(-2, 2),fit_output = False, mode= ['constant','edge']),
                                iaa.TranslateY(px=(-2, 2),fit_output = False, mode= ['constant','edge'])
                                ]),
                            iaa.Rotate(rotate = (-3, 3),fit_output = False, mode= ['constant','edge']), 
                            per_channel=0.5),
            # Зашумление
            iaa.AddElementwise((-40, 40), per_channel = 0.5), # зашумление пикселей случайным добавлением
            iaa.AdditiveGaussianNoise(scale = (0.1*255, 0.2*255), per_channel = 0.5), # зашумление каналов
            iaa.AdditiveLaplaceNoise(scale = (0.05*255, 0.1*255), per_channel = 0.5), # зашумление каналов по Лапласу
            iaa.AdditivePoissonNoise(lam = (0, 30),per_channel = 0.5), # зашумление по Пуассону
            iaa.imgcorruptlike.ImpulseNoise(severity = (1, 2)), # ипульсный шум
            iaa.imgcorruptlike.SpeckleNoise(severity = (1, 2)), # еще один вариант шума
            iaa.SaltAndPepper(p = (0.01, 0.05), per_channel = 0.5), # добавление случаных пикселей
            iaa.imgcorruptlike.Snow(severity = 1), # снег (добавление пятен)
            iaa.imgcorruptlike.Spatter(severity = (1, 2)), # добавление пятен-бликов
            # Функции по внесению новых данных или исключению старых
            iaa.Cutout(nb_iterations = 128, size = (0.01, 0.02), squared=False, 
                       fill_mode=("gaussian","constant"), cval = (0, 255), 
                       fill_per_channel = 0.5),
            iaa.Dropout(p = (0.01, 0.10), per_channel = 0.5), # dropout пиксели
            iaa.CoarseDropout(p=(0.01, 0.02), size_percent=(0.1, 0.2), per_channel=0.75),
            # Функции по размытию и фильтрации
            iaa.GaussianBlur(sigma=(0.1, 0.5)), # размытие
            iaa.imgcorruptlike.MotionBlur(severity = 1), # размытие в движении
            iaa.JpegCompression(compression = (50, 90)), # размытие от компрессии
            iaa.imgcorruptlike.Pixelate(severity = (2, 3)), # пикселизация
            iaa.pillike.EnhanceSharpness(factor = (2, 4)),
            iaa.pillike.EnhanceColor(),
            iaa.pillike.FilterSharpen(), # увеличение резкости
            iaa.pillike.FilterDetail(),
            iaa.Rain(speed=(0.1, 0.3)),
            ],random_order = True),
        iaa.SomeOf((1,None),[
            iaa.BlendAlphaHorizontalLinearGradient(iaa.Add((-80, 10))),
            iaa.BlendAlphaVerticalLinearGradient(iaa.Add((-80, 10)))
            ],random_order = True),
        ])
    ])
'''
# для отладки
img = cv2.cvtColor(cv2.imread('./tests/28807.jpg'), cv2.COLOR_BGR2RGB)
# корректировка размера и формата
if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
if img.shape[-1] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
img = aug_img(image = img)

#img = aug_mp(img)


plt.imshow(img)
plt.show()
'''

# Аугментация 1 изображения WxHx3 и маски WxHx1
def aug_rgb_images(images):
    images = np.uint8(images)
    images = aug_img(images = images)
    return images

# загрузка изображения
def load_img(img_file):
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    # корректировка размера и формата
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[-1] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

# возвращает число батчей в наборе данных
def get_num_batches(data_t, batch_size):
    return int(len(data_t)/batch_size)

# генератор данных для обучения/валидации
def generate_batch(data_t, # папка с вырезанными картинами
                   u_classes,
                   shuffle = True,
                   aug = True,
                   img_size = 320,
                   batch_size = 8):
    while True:
        if shuffle: data_t = np.random.permutation(data_t)
        # получить число батчей в выборке
        num_batches = get_num_batches(data_t, batch_size)
        if num_batches <= 0:
            sys.exit("Ошибка в таблице данных!")
        for i in range(num_batches):
            # получить пути до изображений
            data_batch = data_t[i*batch_size:(i+1)*batch_size]
            # сформировать батчи изображений и меток к ним
            X = np.zeros((batch_size, img_size, img_size, 3), dtype = np.uint8)
            y = np.zeros((batch_size, len(u_classes)), dtype = np.float32)
            for j in range(batch_size):
                # загрузка изображения (картины)
                img_file = data_batch[j, 0]
                #print(img_file)
                img = load_img(img_file)
                # пересжатие до рабочего изобрадения картины: 15%-90%
                img = cv2.resize(img, (img_size, img_size))               
                # аугентация
                X[j] = img
                y[j] = np.float32(u_classes == data_batch[j, 1])
            if aug:
                X = aug_rgb_images(X)
            X = np.float32(X)
            yield X, y

'''
# отладка
import time
train_data = np.loadtxt('D:/1Personal/F/2024/03/nikewalce1/New Plant Diseases Dataset(Augmented)/train.txt', dtype = str, delimiter = '\t')
u_classes = np.loadtxt('D:/1Personal/F/2024/03/nikewalce1/New Plant Diseases Dataset(Augmented)/u_classes.txt', dtype = str, delimiter = '\t')

# генератор
g = generate_batch(data_t = train_data,
                   u_classes = u_classes,
                   img_size = 224,
                   shuffle = True, 
                   aug = True, 
                   batch_size = 4)


X, y = next(g)
'''

'''
num_train_steps = 10000
#st = time.time()
for i in range(num_train_steps):
    X, y = next(g)
    #plt.imshow(np.uint8(X[0]))
    #plt.show()
    #print(y)
    print(i)
'''

# Визуализация тренировочного процесса
def viz_train_val(train_loss,
                  train_ac,
                  val_loss,
                  val_ac,
                  out_file = os.getcwd()+'/train_val_plots.png'):
    fig, axs = plt.subplots(2, 2, constrained_layout = True)
    fig.suptitle('Тренировочные метрики и потери')
    axs[0,0].plot(train_loss, color = "b")
    axs[0,0].set_title('training loss')
    axs[0,0].set(xlabel='epoches', ylabel='loss')
    axs[0,0].grid()
    axs[0,0].xaxis.set_major_locator(MaxNLocator(integer = True))

    axs[0,1].plot(val_loss, color="b")
    axs[0,1].set_title('validation loss')
    axs[0,1].set(xlabel='epoches', ylabel='loss')
    axs[0,1].grid()
    axs[0,1].xaxis.set_major_locator(MaxNLocator(integer = True))

    axs[1,0].plot(train_ac, color="r")
    axs[1,0].set_title('training mAc')
    axs[1,0].set(xlabel='epoches', ylabel='mAc')
    axs[1,0].grid()
    axs[1,0].xaxis.set_major_locator(MaxNLocator(integer = True))

    axs[1,1].plot(val_ac, color="r")
    axs[1,1].set_title('validation mAc')
    axs[1,1].set(xlabel='epoches', ylabel='mAc')
    axs[1,1].grid()
    axs[1,1].xaxis.set_major_locator(MaxNLocator(integer = True))

    plt.savefig(out_file, dpi = 300)
    #plt.show()
    return None

#viz_train_val(train_loss, train_ac, val_loss, val_ac) # сохраняем визуализацию в файл





