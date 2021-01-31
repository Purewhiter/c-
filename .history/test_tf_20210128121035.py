'''
Author      : PureWhite
Date        : 2021-01-28 07:53:43
LastEditors : PureWhite
LastEditTime: 2021-01-28 12:10:35
Description : 
'''
import tensorflow as tf

#查看tensorflow版本
print(tf.__version__)
print('GPU', tf.test.is_gpu_available())
tf.config.list_physical_devices('GPU')