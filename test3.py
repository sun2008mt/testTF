import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 加载数据
# one_hot=True表示对label进行one-hot编码，比如标签4可以表示为[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]，
# 这是神经网络输出层要求的格式
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 定义超参数和placeholder
# 超参数
learning_rate = 0.5
epochs = 10
batch_size = 100

# placeholder
# 输入图片为28 * 28像素 = 784
x = tf.placeholder(tf.float32, [None, 784])
# 输出为0-9的one-hot编码
y = tf.placeholder(tf.float32, [None, 10])

# 定义参数w和b(tf.random_normal()生成正态分布的随机数)
# hidden layer => w, b
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300], name='b1'))
# output layer => w, b
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# 构造隐层网络
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# 构造输出(预测值)(对于单标签多分类任务，输出层的激活函数都是tf.nn.softmax())
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

# 定义loss(交叉熵)
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(
    tf.reduce_sum(
        y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# 定义优化算法
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy)

# 定义初始化operation和准确率node
init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练
with tf.Session() as sess:
    # 变量初始化
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run(
                [optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
    print(sess.run(
        accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))