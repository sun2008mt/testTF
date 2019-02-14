import tensorflow as tf
import numpy as np

# 定义常量
const = tf.constant(2.0, name='const')

# 定义变量(None表示不确定)
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, dtype=tf.float32, name='c')

# 定义运算（运算函数）
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# 定义初始化运算(TensorFlow中所有的变量必须经过初始化才能使用)
init_op = tf.global_variables_initializer()

# 创建会话
with tf.Session() as sess:
    # 运行初始化运算
    sess.run(init_op)
    # 计算
    a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print('Variable a is {}'.format(a_out))
