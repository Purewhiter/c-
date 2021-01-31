'''
Author      : PureWhite
Date        : 2021-01-28 07:53:43
LastEditors : PureWhite
LastEditTime: 2021-01-28 08:22:14
Description : 
'''
import tensorflow as tf
# a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用
# b = tf.test.is_gpu_available()
# print(a,b)
# tf.config.list_physical_devices('GPU')
sess = tf.Session() 
a = tf.constant(1) 
b = tf.constant(2) 
print(sess.run(a+b)) 