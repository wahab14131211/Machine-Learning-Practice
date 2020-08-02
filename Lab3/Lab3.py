# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def TrainModel(num_of_layers,neurons_per_layer,optimizer):
    global train_labels,train_images, test_labels,test_images, Results
    # Build and compile the initial model
    model_array = [keras.layers.Flatten(input_shape=(28, 28))]
    for i in range(num_of_layers):
        model_array.append(keras.layers.Dense(neurons_per_layer, activation='relu'))
    model_array.append(keras.layers.Dense(10))
    model = keras.Sequential(model_array)
    del model_array
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # Train the model
    model.fit(train_images, train_labels, epochs=10)

    # Evaluate Accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    #print("\nTest accuracy for {} Hidden layers with {} Neurons per layer with {} optimizer is: {}".format(num_of_layers,neurons_per_layer,optimizer,test_acc))
    Results.append("Test accuracy for {} Hidden layers with {} Neurons per layer with {} optimizer is: {}".format(num_of_layers,neurons_per_layer,optimizer,test_acc))

if __name__ == '__main__':
    print("TensorFlow version is: {}".format(tf.__version__))

    #import MNIST dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    #EDA
    print("Shape of training images is: {}".format(train_images.shape))
    print("Number of labels in training set is: {}".format(len(train_labels)))
    print("The possible labels are: {}".format(train_labels))
    print("The shape of the test_images is: {}".format(test_images.shape))
    print("The length of the test set is: {}".format(len(test_labels)))

    #Show first image in training set
    #plt.figure()
    #plt.imshow(train_images[0])
    #plt.colorbar()
    #plt.grid(False)
    #plt.show()

    #Preform MinMax normalization on images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #Show first 25 images in training set
    #plt.figure(figsize=(10,10))
    #for i in range(25):
    #    plt.subplot(5,5,i+1)
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.grid(False)
    #    plt.imshow(train_images[i], cmap=plt.cm.binary)
    #    plt.xlabel(class_names[train_labels[i]])
    #plt.show()

    Results = list()
    TrainModel(1,128, 'adam')
    TrainModel(5,20, 'adam')
    TrainModel(2, 50, 'adam')
    TrainModel(5, 200, 'adam')
    TrainModel(10, 100, 'adam')
    TrainModel(5, 1000, 'adam')
    TrainModel(10, 500, 'adam')
    TrainModel(1,128, 'sgd')
    TrainModel(5,20, 'sgd')
    TrainModel(2, 50, 'sgd')
    TrainModel(5, 200, 'sgd')
    TrainModel(10, 100, 'sgd')
    TrainModel(5, 1000, 'sgd')
    TrainModel(10, 500, 'sgd')

    #Use CNN
    # Load the fashion-mnist pre-shuffled train data and test data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]

    # Reshape input data from (28, 28) to (28, 28, 1)
    w, h = 28, 28
    x_train = x_train.reshape(x_train.shape[0], w, h, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
    x_test = x_test.reshape(x_test.shape[0], w, h, 1)

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_valid = tf.keras.utils.to_categorical(y_valid, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = tf.keras.Sequential()

    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    Results.append("Test accuracy for CNN is {}".format(test_acc))


    for result in Results:
        print(result)