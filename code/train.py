#from __future__ import absolute_import, division, print_function
import tensorflow as tf
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    EPOCHS, BATCH_SIZE, save_model_dir, save_every_n_epoch
import math
from models import resnext
from preprocessing import *

#https://github.com/calmisential/Basic_CNNs_TensorFlow2

#EPOCHS = 2
#BATCH_SIZE = 1

#IMAGE_HEIGHT = 240
#IMAGE_WIDTH = 320
#CHANNELS = 3

def generate_datasets(count):
    x, y, count = prepare_data(count)
    tot_size = x.shape[0]
    train_x = x[:int(0.8*tot_size)]
    train_y = y[:int(0.8*tot_size)]
    val_x = x[int(0.8*tot_size):int(0.9*tot_size)]
    val_y = y[int(0.8*tot_size):int(0.9*tot_size)]
    test_x = x[int(0.9*tot_size):]
    test_y = y[int(0.9*tot_size):]

    train_count = train_x.shape[0]
    valid_count = val_x.shape[0]
    test_count = test_y.shape[0]

    return train_x, train_y, val_x, val_y, test_x, test_y, train_count, valid_count, test_count, count

def get_model():
        return resnext.ResNeXt101()


def print_model_summary(network):
    network.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # create model
    model = get_model()
    print_model_summary(network=model)

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.RMSprop()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    # @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(y_true=label_batch, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # start training
    for epoch in range(EPOCHS):
        # get the dataset
        count = 0
        while (count != -1):
            train_x, train_y, val_x, val_y, test_x, test_y, train_count, valid_count, test_count, count = generate_datasets(count)

            step = 0
            for i in range(0, train_x.shape[0], BATCH_SIZE):
                step += 1
                images = train_x[i:i+BATCH_SIZE]
                labels = train_y[i:i+BATCH_SIZE]
                train_step(images, labels)
                print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch,
                                                                                        EPOCHS,
                                                                                        step,
                                                                                        math.ceil(train_count / BATCH_SIZE),
                                                                                        train_loss.result().numpy(),
                                                                                        train_accuracy.result().numpy()))

            for i in range(0, val_x.shape[0], BATCH_SIZE):
                valid_images = val_x[i:i+BATCH_SIZE]
                valid_labels = val_y[i:i+BATCH_SIZE]
                valid_step(valid_images, valid_labels)

            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                    EPOCHS,
                                                                    train_loss.result().numpy(),
                                                                    train_accuracy.result().numpy(),
                                                                    valid_loss.result().numpy(),
                                                                    valid_accuracy.result().numpy()))
            train_loss.reset_states()
            train_accuracy.reset_states()
            valid_loss.reset_states()
            valid_accuracy.reset_states()
            train_x, train_y, val_x, val_y, test_x, test_y = None, None, None, None, None, None

        if epoch % save_every_n_epoch == 0:
            model.save_weights(filepath=save_model_dir+"epoch-{}".format(epoch), save_format='tf')


    # save weights
    model.save_weights(filepath=save_model_dir+"model", save_format='tf')

    # save the whole model
    # tf.saved_model.save(model, save_model_dir)

    # convert to tensorflow lite format
    # model._set_inputs(inputs=tf.random.normal(shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)
