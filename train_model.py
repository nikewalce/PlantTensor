#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Загрузка необходимых библиотек
import tensorflow as tf
tf.keras.backend.clear_session()

# Обучение в 16 битном режиме
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import argparse, pickle, time, os
import numpy as np
import data_utils as du

from  tensorflow.keras.callbacks import Callback, TerminateOnNaN
from tensorflow.keras.optimizers import Adam as optimizer
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.losses import CategoricalCrossentropy as loss_function

from en_lite import EfficientNetLiteB0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Параметры командной строки
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Запуск с параметрами:")  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # число эпох обучения
    parser.add_argument('--num_epochs', type = int, default = 45)
    # файл для сохранения весов лучшей модели
    parser.add_argument('--best_weights', type = str, default = './best_weights.h5')
    # файл для сохранения весов модели на последней итерации обучения
    parser.add_argument('--last_weights', type = str, default = './last_weights.h5')
    # файл визуализации тренировки
    parser.add_argument('--viz_file', type = str, default = './train_val_learn.png')
    # начальный коэффициент обучения
    parser.add_argument('--init_lr', type = float, default = 1e-3)
    # финальный итоговый коэффициент обучения
    parser.add_argument('--final_lr', type = float, default = 1e-6)   
    # папка с исходными данными
    parser.add_argument('--source_folder', type = str, default = './New Plant Diseases Dataset(Augmented)')
    # размер батча для тренировки и валидации
    parser.add_argument('--batch_size', type = int, default = 32)
    # размер изображения
    parser.add_argument('--image_size', type = int, default = 224)
    # парсинг параметров        
    opt = parser.parse_args()

print(opt)
time.sleep(0.1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Основной процесс тренировки
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# получить список всех классов
u_classes = np.loadtxt(opt.source_folder+'/u_classes.txt', dtype = str, delimiter = '\t')
print("Число классов: {}".format(len(u_classes)))

train_data = np.loadtxt(opt.source_folder+'/train.txt', dtype = str, delimiter = '\t')
val_data = np.loadtxt(opt.source_folder+'/val.txt', dtype = str, delimiter = '\t')

# Инициализация нейронной сети
m = EfficientNetLiteB0(input_shape = (opt.image_size, opt.image_size, 3), 
                       return_logits = True,
                       n_classes = len(u_classes))

# Загрузим последние веса модели, если они были ранее получены
if opt.last_weights is not None:
    if os.path.exists(opt.last_weights):
        m.load_weights(opt.last_weights, by_name = True, skip_mismatch = True)
        print("Веса модели загружены!")
    else:
        print("Веса модели не найдены!")
# выведем информацию о модели и её параметрах
trainableParams = np.int32(np.sum([np.prod(v.get_shape()) for v in m.trainable_weights]))
nonTrainableParams = np.int32(np.sum([np.prod(v.get_shape()) for v in m.non_trainable_weights]))
totalParams = trainableParams + nonTrainableParams
print("Модель инициализирована!")
print('Число тренируемых параметров: {}'.format(trainableParams))
print('Число нетренируемых параметров: {}'.format(nonTrainableParams))
print('Общее число параметров: {}'.format(totalParams))

#print(m.summary()) # вывод структуры модели
m.compile(loss = loss_function(from_logits = True, label_smoothing = 0.01),
          optimizer = optimizer(learning_rate = opt.init_lr),
          metrics = ['categorical_accuracy'])

# Определение генераторов данных для тренировки и валидации
train_gen = du.generate_batch(data_t = train_data,
                              u_classes = u_classes,
                              img_size = opt.image_size,
                              shuffle = True, 
                              aug = True, 
                              batch_size = opt.batch_size)

val_gen = du.generate_batch(data_t = val_data,
                              u_classes = u_classes,
                              img_size = opt.image_size,
                              shuffle = True, 
                              aug = False, 
                              batch_size = opt.batch_size)

# расчет числа батчей для каждого дейта-сета
num_train_steps = du.get_num_batches(data_t = train_data, batch_size = opt.batch_size)
num_val_steps = du.get_num_batches(data_t = val_data, batch_size = opt.batch_size)

print("Число файлов для тренировки модели: {}".format(len(train_data)))
print("Число файлов для валидации модели: {}".format(len(val_data)))

# Определение callbacks
# сохранение лучшего значения loss и весов модели
class base_callback(Callback):
    def __init__(self):
        # определение имени файла в котором хранится наилучшее значение Accuracy
        self.file_best_loss = 'best_loss.pkl'
        # попытка загрузки этого значения
        self.best_loss = np.Inf
        try:
            with open(self.file_best_loss, 'rb') as f:
                self.best_loss = pickle.load(f)
                print("Установлено начальное значение loss: {:1.5f}".format(self.best_loss))
        except Exception:
            print("Последнее значение loss не найдено!") 
        # инициализация алгоритма уменьшения learning rate
        self.lr_scheduler = CosineDecayRestarts(initial_learning_rate = opt.init_lr, 
                                                first_decay_steps = opt.num_epochs//3, 
                                                m_mul = 0.5,
                                                alpha = opt.final_lr) 

    # установление коэффициента обучения перед началом новой эпохи
    def on_epoch_begin(self, epoch, logs = None):
        # расчет и установка коэффициента обучения
        m.optimizer.learning_rate = self.lr_scheduler(epoch)
        print("Текущий коэффициент обучения: {:1.7f}".format(m.optimizer.learning_rate.numpy()))
        
    # функция сохранения текущих весов модели
    def on_epoch_end(self, epoch, logs=None):
        # схохранение текущих весов
        m.save_weights(opt.last_weights)
        # текущая метрика по валидации
        c_val_loss = logs['val_loss']
        if self.best_loss > c_val_loss:
            print('\nВалидационный loss уменьшился с {:1.5f} до {:1.5f}, сохраняем модель {}'.
                  format(self.best_loss, c_val_loss, opt.best_weights))
            self.best_loss = c_val_loss # перезаписываем наилучшую метрику
            # сохранение наилучших весов
            m.save_weights(opt.best_weights)
            # сохранение наилучшего значения loss
            with open(self.file_best_loss, 'wb') as f:
                pickle.dump(self.best_loss, f)

cb_base = base_callback() # инициализация базового callback
cb_tnan = TerminateOnNaN() # завершение работы, если случился nan-loss

# Тренировка и валидация
history = m.fit(x = train_gen,
                batch_size = opt.batch_size,
                epochs = opt.num_epochs,
                callbacks = [cb_base, cb_tnan],
                validation_data = val_gen,
                steps_per_epoch = num_train_steps,
                validation_steps = num_val_steps,
                verbose = 1)

# визуализация
train_loss, train_iou, _, val_loss, val_iou, _ = list(history.history.values())
du.viz_train_val(train_loss, train_iou, val_loss, val_iou)
print("Нейронная сеть обучена!")
