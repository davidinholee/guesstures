import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.keras.layers import BatchNormalization, Flatten
from preprocessing import *
import numpy as np

print(tf.__version__)


weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1
cardinality = 8 #32 #8 # how many split ?
blocks = 3 # res_block ! (split + transition)

"""
The total number of layers is (3*blocks)*residual_layer_num + 2, because:
- blocks = split(conv 2) + transition(conv 1) = 3 layer
- first conv layer 1, last dense layer 1
"""

depth = 64 # out channel

batch_size = 128
iteration = 391
# 128 * 391 ~ 50,000

test_iteration = 10

total_epochs = 300

def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_Pooling(in_put, stride=2, padding='SAME', name="max_pool"):
    return tf.nn.max_pool(input=in_put, strides=stride)

def BatchNormalizationalization(x, training, scope):
    return tf.cond(training,
        lambda : BatchNormalization(inputs=x, is_training=training, reuse=None,
                scope=scope,
                updates_collections=None,
                decay=0.9,
                center=True,
                scale=True,
                zero_debias_moving_mean=True),
        lambda : BatchNormalization(inputs=x, is_training=training, reuse=True,
                scope=scope,
                updates_collections=None,
                decay=0.9,
                center=True,
                scale=True,
                zero_debias_moving_mean=True))

def Relu(x):
    return tf.nn.relu(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, use_bias=False, units=class_num, name='linear')

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration # average loss
    test_acc /= test_iteration # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary

class ResNeXt():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_ResNext(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
            x = BatchNormalizationalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def conv_batch(self, x, filter_num, stride_num, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=filter_num, kernel=[1,1], stride=stride_num, layer_name=scope+'_conv1')
            x = BatchNormalizationalization(x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=depth, kernel=[1,1], stride=stride, layer_name=scope+'_conv1')
            x = BatchNormalizationalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
            x = BatchNormalizationalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, stride, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = BatchNormalizationalization(x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(cardinality) :
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def dense_layer(self, unit_num, class_num, scope):
        with tf.name_scope(scope):
            x = Dense(x, units=class_num, layer_name=scope+'_dense1')
        return x

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge

        for i in range(res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))

            if flag is True :
                pad_input_x = Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else :
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return input_x


    def Build_ResNext(self, input_x):

        #Resnext101
        #modelled after https://github.com/ahmetgunduz/Real-time-GesRec/blob/master/models/resnext.py
        #changed convolution layers not 3d

        input_x = self.first_layer(input_x, scope='first_layer')
        #x = self.conv3d(64, kernel_size=7, strides=(1, 2, 2), padding=(3, 3, 3), bias=false)
        #x = self.bn1(x)
        #x = self.relu(x)
        x = self.Max_Pooling(input_x)

        x = self.conv_batch(x, filter_num=128, stride_num=1, scope='conv1')
        #x = self.layer1(x)
        x = self.conv_batch(x, filter_num=256, stride_num=2, scope='conv2')
        #x = self.layer2(x)
        x = self.conv_batch(x, filter_num=512, stride_num=2, scope='conv3')
        #x = self.layer3(x)
        x = self.conv_batch(x, filter_num=1024, stride_num=2, scope='conv4')
        #x = self.layer4(x)
        x = Global_Average_Pooling(x)
        #x = self.avgpool(x)
        x = x.reshape(x, (x.size(0), -1))
        #x = x.view(x.size(0), -1)
        x = Linear(x)
        # x = self.tf.dense(x)

        #Resnext bottleneck
        residual = x
        x = self.first_layer(x, scope='first_layer_num2')
        #x = self.conv1(input_x)
        #x = self BatchNormalization(x)
        #x = self.relu(x)
        x = self.conv_batch(x, filter_num=256, stride_num=2, scope='bottlenextconv2')
        #x = self.conv2(x)
        #x = self BatchNormalization(x)
        x = Relu(x)
        #x = self.relu(x)

        x = self.conv_batch(x, filter_num=512, stride_num=2, scope='bottlenextconv3')
        #x = self.conv3(x)
        #x = self.batchnorm()

        x += residual
        x = Relu(x)
        #x = self.relu(x)

        #resnext_regular_architecture
        #https://github.com/taki0112/ResNeXt-Tensorflow/blob/60bfd72c5c944ca960f2c906406772c8901cdcef/ResNeXt.py#L37
        #input_x = self.first_layer(input_x, scope='first_layer')

        #x = self.residual_layer(input_x, out_dim=64, layer_num='1')
        #x = self.residual_layer(x, out_dim=128, layer_num='2')
        #x = self.residual_layer(x, out_dim=256, layer_num='3')

        #x = Global_Average_Pooling(x)
        #x = Flatten(x)
        #x = Linear(x)

        return x


if __name__ == "__main__":

    print("Beginning preprocesing...")
    x, y = prepare_data()
    train_x = x
    train_y = y
    test_x = x
    test_y = y
    print("Preprocessing finished!")

    # IMAGE_SIZE = 32, IMAGE_CHANNELS = 3, CLASS_NUM = 10 in cifar10 (CHECK PREPROCESSING FOR TRUE IMAGE SIZES)
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS])
    label = tf.placeholder(tf.float32, shape=[None, CLASS_NUM])

    training_flag = tf.placeholder(tf.bool)


    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    logits = ResNeXt(x, training=training_flag).model
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
    train = optimizer.minimize(cost + l2_loss * weight_decay)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./logs', sess.graph)

        epoch_learning_rate = init_learning_rate
        for epoch in range(1, total_epochs + 1):
            if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
                epoch_learning_rate = epoch_learning_rate / 10

            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0

            for step in range(1, iteration + 1):
                if pre_index + batch_size < 50000:
                    batch_x = train_x[pre_index: pre_index + batch_size]
                    batch_y = train_y[pre_index: pre_index + batch_size]
                else:
                    batch_x = train_x[pre_index:]
                    batch_y = train_y[pre_index:]

                batch_x = data_augmentation(batch_x)

                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size


            train_loss /= iteration # average loss
            train_acc /= iteration # average accuracy

            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                            tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

            test_acc, test_loss, test_summary = Evaluate(sess)

            summary_writer.add_summary(summary=train_summary, global_step=epoch)
            summary_writer.add_summary(summary=test_summary, global_step=epoch)
            summary_writer.flush()

            line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
            print(line)

            with open('logs.txt', 'a') as f:
                f.write(line)

            saver.save(sess=sess, save_path='./model/ResNeXt.ckpt')