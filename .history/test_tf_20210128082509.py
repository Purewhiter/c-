'''
Author      : PureWhite
Date        : 2021-01-28 07:53:43
LastEditors : PureWhite
LastEditTime: 2021-01-28 08:25:08
Description : 
'''
import tensorflow as tf
# tf.config.list_physical_devices('GPU')
import tensorflow as tf

#查看tensorflow版本
print(tf.__version__)

print('GPU', tf.test.is_gpu_available())

a = tf.constant(2.0)
b = tf.constant(4.0)
print(a + b)