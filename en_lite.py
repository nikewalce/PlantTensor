import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, Activation, Add, Softmax
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D, Dropout, Dense, ReLU
from tensorflow.python.keras.layers import Normalization
from tensorflow.keras.layers import GlobalMaxPooling2D, Flatten, Multiply, Reshape, Softmax
from tensorflow.python.keras.layers import Rescaling
from tensorflow.keras.models import Model
#from tensorflow_addons.activations import sparsemax
#from tensorflow_addons.layers import Sparsemax
from tensorflow.keras.layers import Layer

import tensorflow.keras.backend as K
import numpy as np


# https://github.com/Jongchan/attention-module/blob/5d3a54af0f6688bedca3f179593dff8da63e8274/MODELS/cbam.py#L26
# simple mpl
def mlp(n_inputs, n_outputs, act_f = 'relu6'):
    inputs = Input(shape = n_outputs)
    x = Dense(units=n_inputs)(inputs)
    if act_f=='relu6':
        x = ReLU(max_value=6.0)(x)
    else:
        x = ReLU()(x)
    x = Dense(units=n_outputs)(x)
    model = Model(inputs = inputs, outputs = x)
    #print(model.summary())
    return model

# Normalization layer
'''
class Normalization(Layer):
    def __init__(self, means, stds, trainable = False):
        super(Normalization, self).__init__()
        self.means = tf.Variable(initial_value = means, trainable = False, dtype = tf.float32, name = 'Means_of_Images')
        self.stds = tf.Variable(initial_value = stds, trainable = False,dtype = tf.float32, name = 'Stds_of_Images')
    def call(self,x):
        x = tf.math.subtract(x,self.means)
        x = tf.math.divide_no_nan(x,self.stds)
        return x
'''
#x = np.ones((2,3,3,3),np.float32)
#x[1] = x[1]*3
#y = Normalization(means = [1,2,3],stds = [1,1,3])(x)  
    
# attention gate
def channel_gate(x, reduction_ratio=32):
    x1 = GlobalAveragePooling2D()(x)
    x2 = GlobalMaxPooling2D()(x)
    m =  mlp(n_inputs = x.shape[-1]//reduction_ratio,n_outputs = x.shape[-1])
    x1 = m(x1)
    x2 = m(x2)
    y = x1+x2
    y = Activation('sigmoid')(y)
    #y = Reshape(target_shape = (1,1,y.shape[-1]))(y)
    x = Multiply()([x,y])
    return x
 

#y = np.array([[1,2,3,4]],np.float32)
#r = Multiply()([x,y])
#import numpy as np
#x = np.ones((1,10,10,1280))
#y = channel_gate(x)    
#print(y.shape)

# Generalized Mean Pooling: x>0, p>0 !
# https://paperswithcode.com/method/generalized-mean-pooling
class GeneralizedMeanPooling(Layer):
    def __init__(self, trainable = True):
        super(GeneralizedMeanPooling, self).__init__()        
        self.p = tf.Variable(initial_value=1.0,
                             constraint = lambda x: tf.clip_by_value(x, K.epsilon(), 3.0),
                             trainable=True, name = 'Generalized_Mean_Pooling_Degree')
    def call(self, inputs):
        inputs = inputs+K.epsilon()
        inputs = tf.pow(inputs,self.p) # exponentiation
        inputs = tf.reduce_mean(inputs,axis=(1,2)) # global average pooling
        inputs = tf.pow(inputs,1./self.p) # exponentiation
        return inputs 

#x = tf.ones((1,4,4,3),dtype=np.float32)+3.
#x = GeneralizedMeanPooling()(x)

# Функция расширения слоя на число фильтров
def round_filters(filters, width_coefficient, depth_divisor):    
    filters *= width_coefficient
    new_filters = int(filters+depth_divisor/2)//depth_divisor*depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Удостовериться, что число фильтров сократилось не более чем на 90%.
    if new_filters<0.9*filters:
        new_filters+= depth_divisor    
    return int(new_filters)

# функция вычисляющая число слоёв
def round_repeats(repeats, depth_coefficient):
    return int(np.ceil(depth_coefficient*repeats))

# Основная часть Mobile inverted bottlenec блока
def mb_conv(inputs, in_channels, expansion_factor, 
            k, stride, out_channels, drop_connect_rate,
            act_f = 'relu6', kernel_regularizer=None):
    # Расширение в глубину
    if expansion_factor!=1:
        x = Conv2D(filters=in_channels*expansion_factor,
                   kernel_size=(1, 1),
                   strides=1,
                   padding="same",
                   use_bias=False,
                   kernel_regularizer=kernel_regularizer)(inputs)
        x = BatchNormalization()(x)
        if act_f=='relu6':
            x = ReLU(max_value=6.0)(x)
        else:
            x = ReLU()(x)
    else:
        x = inputs    
    x = DepthwiseConv2D(kernel_size=(k,k),
                        strides=stride,
                        padding="same",
                        use_bias=False,
                        kernel_regularizer=kernel_regularizer) (x)
    x = BatchNormalization()(x)
    if act_f=='relu6':
        x = ReLU(max_value=6.0)(x)
    else:
        x = ReLU()(x)
    x = Conv2D(filters=out_channels,
               kernel_size=(1, 1),
               strides=1,
               padding="same",
               use_bias=False,
               kernel_regularizer=kernel_regularizer) (x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x) # у официалов нет...
    # Drop-connection
    if stride == 1 and in_channels == out_channels:
        if drop_connect_rate:
            x = Dropout(rate=drop_connect_rate)(x)
        x = Add()([x, inputs])
    return x

# Построение Mobile inverted bottlenec блока
def build_mb_conv_block(inputs, in_channels, out_channels, layers, 
                        stride, expansion_factor, k, drop_connect_rate,
                        act_f = 'relu6',kernel_regularizer=None,
                        fine_tune = False):
    # Первый элемент блока
    x = mb_conv(inputs = inputs, in_channels=in_channels,out_channels=out_channels,
                        expansion_factor=expansion_factor,stride=stride,
                        k=k,drop_connect_rate=drop_connect_rate, act_f = act_f,
                        kernel_regularizer=kernel_regularizer)
    # Последующие
    if layers>1:
        for i in range(layers-1):
            x = mb_conv(inputs = x, in_channels=out_channels,
                        out_channels=out_channels,
                        expansion_factor=expansion_factor,
                        stride=(1,1),k= k,
                        drop_connect_rate=drop_connect_rate,
                        act_f = act_f,
                        kernel_regularizer=kernel_regularizer)
    return x

# Основная сеть. По умолчанию EfficientNet - B0
def EfficientNetLite(input_shape = (224,224,3), 
                     width_coefficient = 1.,
                     depth_coefficient = 1., 
                     dropout_rate = 0.2, 
                     drop_connect_rate=0.2, 
                     depth_divisor=8,
                     include_top = True, 
                     n_classes = 1000,
                     act_f = 'relu6',
                     pool = 'avg',
                     n_final_filters=1280,
                     return_logits = False,
                     top_activation = 'softmax',
                     means = None,
                     stds = None,
                     kernel_regularizer=None):
    # Определение входов сети
    inputs = Input(shape = input_shape)
    # Масштабирование
    x = Rescaling(scale=1./255, offset=0)(inputs)
    # Нормализация
    if means is None: means = [0.,0.,0.]
    if stds is None: stds = [1.,1.,1.]
    x = Normalization(mean = means, variance = list(np.power(stds,2)), axis = 3) (x)
    # Часть по извлечению фич
    x = Conv2D(filters=32,
               kernel_size=(3,3),
               strides=(2,2),
               padding="same", 
               use_bias=False,
               kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    if act_f=='relu6':
        x = ReLU(max_value=6.0)(x)
    else:
        x = ReLU()(x)
    # MB - блоки
    x = build_mb_conv_block(inputs=x, 
                            in_channels=round_filters(32, width_coefficient,depth_divisor),
                            out_channels=round_filters(16, width_coefficient,depth_divisor),
                            layers=1,
                            stride=1,
                            expansion_factor=1, 
                            k=3, 
                            drop_connect_rate=drop_connect_rate,
                            act_f = act_f,
                            kernel_regularizer=kernel_regularizer)
    x = build_mb_conv_block(inputs=x, 
                            in_channels=round_filters(16, width_coefficient,depth_divisor),
                            out_channels=round_filters(24, width_coefficient,depth_divisor),
                            layers=round_repeats(2, depth_coefficient),
                            stride=2,
                            expansion_factor=6, 
                            k=3, 
                            drop_connect_rate=drop_connect_rate,
                            act_f = act_f,
                            kernel_regularizer=kernel_regularizer)
    x = build_mb_conv_block(inputs = x, 
                            in_channels=round_filters(24, width_coefficient,depth_divisor),
                            out_channels=round_filters(40, width_coefficient,depth_divisor),
                            # для lite
                            layers=round_repeats(2, depth_coefficient),
                            stride=2,
                            expansion_factor=6, 
                            k=5, 
                            drop_connect_rate=drop_connect_rate,
                            act_f = act_f,
                            kernel_regularizer=kernel_regularizer)
    x = build_mb_conv_block(inputs = x, 
                            in_channels=round_filters(40, width_coefficient,depth_divisor),
                            out_channels=round_filters(80, width_coefficient,depth_divisor),
                            layers=round_repeats(3, depth_coefficient),
                            stride=2,
                            expansion_factor=6,
                            k=3, 
                            drop_connect_rate=drop_connect_rate,
                            act_f = act_f,
                            kernel_regularizer=kernel_regularizer)
    x = build_mb_conv_block(inputs = x, 
                            in_channels=round_filters(80, width_coefficient,depth_divisor),
                            out_channels=round_filters(112, width_coefficient,depth_divisor),
                            layers=round_repeats(3, depth_coefficient),
                            stride=1,
                            expansion_factor=6,
                            k=5, 
                            drop_connect_rate=drop_connect_rate,
                            act_f = act_f,
                            kernel_regularizer=kernel_regularizer)
    x = build_mb_conv_block(inputs = x, 
                            in_channels=round_filters(112, width_coefficient,depth_divisor),
                            out_channels=round_filters(192, width_coefficient,depth_divisor),
                            layers=round_repeats(4, depth_coefficient),
                            stride=2,
                            expansion_factor=6, 
                            k=5, 
                            drop_connect_rate=drop_connect_rate,
                            act_f = act_f,
                            kernel_regularizer=kernel_regularizer)
    x = build_mb_conv_block(inputs = x, 
                            in_channels=round_filters(192, width_coefficient,depth_divisor),
                            out_channels=round_filters(320, width_coefficient,depth_divisor),                            # для lite
                            layers=1,
                            stride=1,
                            expansion_factor=6, 
                            k=3, 
                            drop_connect_rate=drop_connect_rate,
                            act_f = act_f,
                            kernel_regularizer=kernel_regularizer)    
    # Финальная часть
    x = Conv2D(filters=n_final_filters,
               kernel_size=(1, 1),
               strides=1,
               padding="same",
               use_bias=False,
               kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    if act_f=='relu6':
        x = ReLU(max_value=6.0)(x)
    else:
        x = ReLU()(x)
    if pool == 'avg':
        x = GlobalAveragePooling2D()(x)
    if pool == 'max':
        x = GlobalMaxPooling2D()(x)
    # https://paperswithcode.com/method/generalized-mean-pooling
    if pool == 'gavg':
        x = GeneralizedMeanPooling()(x)
    if pool =='dwc':
        k = K.int_shape(x)[1]
        x = DepthwiseConv2D(kernel_size = k, use_bias = False)(x)
        x = Flatten()(x)
    # https://github.com/sayakpaul/Revisiting-Pooling-in-CNNs/blob/main/SoftPool.ipynb
    if pool == 'sp':
        a = Softmax(axis = (1,2))(x)
        x = Multiply()([x,a])
        x = GlobalAveragePooling2D()(x)
    # Часть классификатора
    if include_top == True:
        x = Dropout(rate=dropout_rate)(x)
        x = Dense(units=n_classes)(x)
        if return_logits == False:
            x = Softmax()(x)
    # модель
    model = Model(inputs = inputs, outputs = x)
    return model

# Классы сетей

# 4652008 - W
def EfficientNetLiteB0(input_shape = (224,224,3), include_top = True, n_classes = 1000,
                         act_f = 'relu6', pool = 'avg', 
                         n_final_filters = 1280,
                         return_logits = False, 
                         top_activation = 'softmax',
                         means = None,
                         stds = None,
                         kernel_regularizer=None):
    m = EfficientNetLite(input_shape = input_shape, 
                         width_coefficient = 1.0,
                         depth_coefficient = 1.0, 
                         dropout_rate = 0.2, 
                         drop_connect_rate=0.2, 
                         depth_divisor=8,
                         include_top = include_top, 
                         n_classes = n_classes,
                         act_f = act_f,
                         pool = pool,
                         means = means,
                         stds = stds,
                         n_final_filters = n_final_filters,
                         return_logits = return_logits,
                         top_activation = top_activation,
                         kernel_regularizer=kernel_regularizer)
    return m

# 5416680 - W
def EfficientNetLiteB1(input_shape = (240,240,3), include_top = True, n_classes = 1000,
                         act_f = 'relu6', pool = 'avg', 
                         n_final_filters = 1280,
                         return_logits = False,
                         top_activation = 'softmax',
                         means = None,
                         stds = None,
                         kernel_regularizer=None):
    m = EfficientNetLite(input_shape = input_shape, 
                         width_coefficient = 1.0,
                         depth_coefficient = 1.1, 
                         dropout_rate = 0.2, 
                         drop_connect_rate=0.2, 
                         depth_divisor=8,
                         include_top = include_top, 
                         n_classes = n_classes,
                         act_f = act_f, 
                         means = means,
                         stds = stds,
                         pool = pool,
                         n_final_filters = n_final_filters,
                         return_logits = return_logits,
                         top_activation = top_activation,
                         kernel_regularizer=kernel_regularizer)
    return m

# 6092072 - W
def EfficientNetLiteB2(input_shape = (260,260,3), include_top = True, n_classes = 1000,
                         act_f = 'relu6', pool = 'avg', 
                         n_final_filters = 1280,
                         return_logits = False,
                         top_activation = 'softmax',
                         means = None,
                         stds = None,
                         kernel_regularizer=None):
    m = EfficientNetLite(input_shape = input_shape, 
                         width_coefficient = 1.1,
                         depth_coefficient = 1.2, 
                         dropout_rate = 0.2, 
                         drop_connect_rate=0.3, 
                         depth_divisor=8,
                         include_top = include_top, 
                         n_classes = n_classes,
                         act_f = act_f,
                         means = means,
                         stds = stds,
                         pool = pool,
                         n_final_filters = n_final_filters,
                         return_logits = return_logits,
                         top_activation = top_activation,
                         kernel_regularizer=kernel_regularizer)
    return m

# 8197096 - W
def EfficientNetLiteB3(input_shape = (280,280,3), include_top = True, n_classes = 1000,
                         act_f = 'relu6', pool = 'avg',
                         n_final_filters = 1280,
                         return_logits = False,
                         top_activation = 'softmax',
                         means = None,
                         stds = None,
                         kernel_regularizer=None):
    m = EfficientNetLite(input_shape = input_shape, 
                         width_coefficient = 1.2,
                         depth_coefficient = 1.4, 
                         dropout_rate = 0.2, 
                         drop_connect_rate=0.3, 
                         depth_divisor=8,
                         include_top = include_top, 
                         n_classes = n_classes,
                         act_f = act_f,
                         means = means,
                         stds = stds,
                         pool = pool,
                         n_final_filters = n_final_filters,
                         return_logits = return_logits,
                         top_activation = top_activation,
                         kernel_regularizer=kernel_regularizer)
    return m

# 13006568 - W
def EfficientNetLiteB4(input_shape = (300,300,3), include_top = True, n_classes = 1000,
                         act_f = 'relu6', pool = 'avg',
                         n_final_filters = 1280,
                         return_logits = False,
                         top_activation = 'softmax',
                         means = None,
                         stds = None,
                         kernel_regularizer=None):
    m = EfficientNetLite(input_shape = input_shape, 
                         width_coefficient = 1.4,
                         depth_coefficient = 1.8, 
                         dropout_rate = 0.2, 
                         drop_connect_rate=0.3, 
                         depth_divisor=8,
                         include_top = include_top, 
                         n_classes = n_classes,
                         act_f = act_f,
                         means = means,
                         stds = stds,
                         pool = pool,
                         n_final_filters = n_final_filters,
                         return_logits = return_logits,
                         top_activation = top_activation,
                         kernel_regularizer=kernel_regularizer)
    return m


# отладка

#m = EfficientNetLiteB0(pool = 'sp')
#m.summary()
'''
from tensorflow_model_optimization.quantization.keras import quantize_model
m = quantize_model(m)

import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(m)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
lite_model = converter.convert()
open('./enlite_b0.tflite','wb').write(lite_model)
'''
#m.summary()
#m.save('enlite_b4.h5')