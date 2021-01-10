import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import scipy.io.arff as arff
import pandas as pd
from RocketImplementation import generate_kernels, apply_kernels
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeClassifierCV
from tensorflow.keras.regularizers import l2, l1


# Function to Load a Single Dataset
def load_single_dataset(directory, seed=4000, mode=1):
    '''
    :param directory: Dataset Directory
    :param seed: Seed to control random shuffling (default: 4000)
    :param mode: Flag to control form of labels, 1 = Categorical (One Hot Encoding), else Integer

    :return data: The data (x) from the dataset
    :return labels: The labels (y) from the dataset, format based on the mode.
    '''
    # Load data from the arff into a pandas dataframe
    data, meta_info = arff.loadarff(directory)
    data = pd.DataFrame(data)
    data = data.sample(frac=1).reset_index(drop=True)

    labels = data.pop(data.iloc[:, -1].name)
    categories = labels.nunique()
    integer_label = 0
    for element in labels.unique():
        labels = labels.replace(element, integer_label)
        integer_label += 1
    labels = labels.to_numpy(dtype=np.int32)
    if mode == 1:
        labels = keras.utils.to_categorical(labels, num_classes=categories)

    # Normalize the data
    data = data.to_numpy()
    data -= np.min(data)
    data /= np.max(data)

    # Split into training and validation
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(data, axis=0)
    rng.shuffle(labels, axis=0)

    return data, labels

# Function to Display Performance Plots
def show_performance(history, num_epochs, dataset_name):
    '''
    :param history: The history as a list object generated as output of keras.fit
    :param num_epochs: Number of epochs the model trained for
    :param dataset_name: The name of the dataset used

    '''

    # Isolate measured parameters
    loss = history[0]
    accuracy = history[5]
    val_loss = history[10]
    val_accuracy = history[15]
    x_axis = np.arange(0, num_epochs)

    # Generate Plots
    plt.figure()
    plt.suptitle('Dataset: '+str(dataset_name))
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, loss, 'red', x_axis, val_loss, 'blue')
    plt.legend(['Training', 'Validation'])
    plt.grid()
    plt.title('Categorical CrossEntropy Loss')
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.plot(x_axis, accuracy, 'red', x_axis, val_accuracy, 'blue')
    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy')
    plt.show()

# Dense Net Inspired Network
def dense_net(num_kernels, cat):
    # Noteable concept: No Activation Functions have been used here (Except for the final softmax). This is
    # inspired by the fact that linear connections are demonstrable performance improvements, refer report for more information.
    '''
    :param num_kernels: The number of ROCKET kernels used
    :param cat: The number of categories to classify into, based on labels in the dataset

    :return model: The Dense-Net model object
    '''

    # Initialize Model Layers
    x_in = layers.Input(shape=(2 * num_kernels, ))
    x = layers.Dropout(0.5)(x_in)
    x = layers.Dense(1024)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(cat, 'softmax')(x)

    # Initialize Model, Optimizer, Loss Function and Metrics
    model = keras.Model(inputs=x_in, outputs=x)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_function = keras.losses.CategoricalCrossentropy()
    metrics = [keras.metrics.TruePositives(name='tp'),
               keras.metrics.FalsePositives(name='fp'),
               keras.metrics.TrueNegatives(name='tn'),
               keras.metrics.FalseNegatives(name='fn'),
               keras.metrics.CategoricalAccuracy(name='acc'),
               keras.metrics.Precision(name='precision'),
               keras.metrics.Recall(name='recall'),
               keras.metrics.AUC(name='auc')]
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    
    # Show model summary
    model.summary()
    return model

# Residual Net Inspired Network
def residual_net(num_kernels, cat):
    '''
    :param num_kernels: The number of ROCKET kernels used
    :param cat: The number of categories to classify into, based on labels in the dataset

    :return model: The Res-Net model object
    '''
    # ResNet Block
    def res_net_block(z_in, filters, multiplier, mode=0):
        '''
        :param z_in: The input to the residually connected block.
        :param filters: The number of filters in each layer of the block.
        :param multiplier: The final layer of the block will have "filters * multiplier".
        :param mode: Flag to decide nature of skip connection. 0 == Convolution Block in Skip Connection. Else Identity Skip Connection.

        :return z: Output of the residual block.
        '''
        # Layer 1
        z = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same',
                          kernel_regularizer=l2(0.005), bias_regularizer=l2(0.01))(z_in)
        z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)
        # Layer 2
        z = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                          kernel_regularizer=l2(0.05), bias_regularizer=l2(0.01))(z)
        z = layers.BatchNormalization()(z)
        z = layers.LeakyReLU()(z)
        # Layer 3
        z = layers.Conv2D(filters=multiplier * filters, kernel_size=1, strides=1, padding='same',
                          kernel_regularizer=l2(0.05), bias_regularizer=l2(0.01))(z)
        z = layers.BatchNormalization()(z)
        # Skip Connection Layer
        if mode == 0:
            z2 = layers.Conv2D(filters=multiplier * filters, kernel_size=1, strides=1, padding='same',
                               kernel_regularizer=l2(0.05), bias_regularizer=l2(0.01))(z_in)
            z2 = layers.BatchNormalization()(z2)
            z = layers.Add()([z, z2])
        else:
            z = layers.Add()([z, z_in])
        
        # Final Activation
        z = layers.LeakyReLU()(z)
        return z

    # Initialize Model Layers
    x_in = layers.Input(shape=(2 * num_kernels, ))
    x = layers.Reshape((200, 100, 1))(x_in)
    x = layers.Conv2D(100, 5, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x1 = res_net_block(x, 200, 2, 0)
    x = res_net_block(x1, 400, 1, 1)
    x = res_net_block(x, 400, 1, 1)
    x = res_net_block(x, 400, 1, 1)
    x = layers.Add()([x, x1])

    x1 = res_net_block(x, 600, 2, 0)
    x = res_net_block(x1, 1200, 1, 1)
    x = res_net_block(x, 1200, 1, 1)
    x = res_net_block(x, 1200, 1, 1)
    x = layers.Add()([x, x1])
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(100, 5, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)

    x1 = res_net_block(x, 300, 2, 0)
    x = res_net_block(x1, 600, 1, 1)
    x = res_net_block(x, 600, 1, 1)
    x = res_net_block(x, 600, 1, 1)
    x = layers.Add()([x, x1])

    x1 = res_net_block(x, 600, 2, 0)
    x = res_net_block(x1, 1200, 1, 1)
    x = res_net_block(x, 1200, 1, 1)
    x = res_net_block(x, 1200, 1, 1)
    x = layers.Add()([x, x1])
    x = layers.Activation('relu')(x)

    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, 'linear')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, 'linear')(x)
    x = layers.Dense(cat, activation='softmax')(x)

    # Initialize Model, Optimizer, Loss Function and Metrics
    model = keras.Model(inputs=x_in, outputs=x)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_function = keras.losses.CategoricalCrossentropy()
    metrics = [keras.metrics.TruePositives(name='tp'),
               keras.metrics.FalsePositives(name='fp'),
               keras.metrics.TrueNegatives(name='tn'),
               keras.metrics.FalseNegatives(name='fn'),
               keras.metrics.CategoricalAccuracy(name='acc'),
               keras.metrics.Precision(name='precision'),
               keras.metrics.Recall(name='recall'),
               keras.metrics.AUC(name='auc')]
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    # Show model summary
    model.summary()
    return model

# Inception Net Inspired Network
def inception_net(num_kernels, cat):
    '''
    :param num_kernels: The number of ROCKET kernels used
    :param cat: The number of categories to classify into, based on labels in the dataset

    :return model: The Inception-Net model object
    '''
    def inception_block(z_in):
        '''
        :param z_in: The input to the Inception Block fed to 6 unique layers
        
        :return z: Output of the inception block
        '''
        z1 = layers.Conv2D(filters=20, kernel_size=3, strides=1, padding='same',
                           kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(z_in)
        z2 = layers.Conv2D(filters=20, kernel_size=7, strides=1, padding='same',
                           kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(z_in)
        z3 = layers.Conv2D(filters=20, kernel_size=11, strides=1, padding='same',
                           kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(z_in)
        z4 = layers.Conv2D(filters=20, kernel_size=15, strides=1, padding='same',
                           kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(z_in)
        z5 = layers.Conv2D(filters=20, kernel_size=19, strides=1, padding='same',
                           kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(z_in)
        z6 = layers.Conv2D(filters=20, kernel_size=1, strides=1, padding='same',
                           kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(z_in)
        z = layers.concatenate([z1, z2, z3, z4, z5, z6])
        z = layers.BatchNormalization()(z)
        z = layers.Activation('relu')(z)
        return z

    x_in = layers.Input(shape=(2 * num_kernels))
    x = layers.Reshape((200, 100, 1))(x_in)

    x = layers.MaxPool2D()(x)

    x1 = inception_block(x)
    x = inception_block(x1)
    x = inception_block(x)
    x = layers.Add()([x1, x])

    x = layers.MaxPool2D()(x)

    x1 = inception_block(x)
    x = inception_block(x1)
    x = inception_block(x)
    x = layers.Add()([x1, x])

    x = layers.MaxPool2D()(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(cat, 'softmax')(x)

    # Initialize Model, Optimizer, Loss Function and Metrics
    model = keras.Model(inputs=x_in, outputs=x)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_function = keras.losses.CategoricalCrossentropy()
    metrics = [keras.metrics.TruePositives(name='tp'),
               keras.metrics.FalsePositives(name='fp'),
               keras.metrics.TrueNegatives(name='tn'),
               keras.metrics.FalseNegatives(name='fn'),
               keras.metrics.CategoricalAccuracy(name='acc'),
               keras.metrics.Precision(name='precision'),
               keras.metrics.Recall(name='recall'),
               keras.metrics.AUC(name='auc')]
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    model.summary()
    return model


def auto_encoder_classifier(num_kernels, cat):
    '''
    :param num_kernels: The number of ROCKET kernels used
    :param cat: The number of categories to classify into, based on labels in the dataset

    :return model: The AutoEncoder model object
    '''
    x_in = layers.Input(shape=(2 * num_kernels, ))
    x = layers.Reshape((200, 100, 1))(x_in)

    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=600, kernel_size=3, strides=2, padding='same',
                      kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=600, kernel_size=3, strides=5, padding='same',
                      kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalMaxPooling2D()(x)
    labels = layers.Dense(cat, 'softmax', name='labels')(x)

    x = layers.Dropout(0.6)(labels)
    x = layers.Dense(128, 'relu')(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(200, 'relu')(x)
    x = layers.Reshape((20, 10, 1))(x)
    x = layers.Conv2DTranspose(filters=300, kernel_size=5, strides=2, padding='same',
                               kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters=300, kernel_size=5, strides=5, padding='same',
                               kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same',
                      kernel_regularizer=l2(0.8), bias_regularizer=l2(0.1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    autogen = layers.Reshape((2 * num_kernels, ), name='autogen')(x)

    model = keras.Model(inputs=x_in, outputs=[labels, autogen])
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_function = {'labels': keras.losses.CategoricalCrossentropy(),
                     'autogen': keras.losses.MeanSquaredError()}
    metrics = [keras.metrics.TruePositives(name='tp'),
               keras.metrics.FalsePositives(name='fp'),
               keras.metrics.TrueNegatives(name='tn'),
               keras.metrics.FalseNegatives(name='fn'),
               keras.metrics.CategoricalAccuracy(name='acc'),
               keras.metrics.Precision(name='precision'),
               keras.metrics.Recall(name='recall'),
               keras.metrics.AUC(name='auc')]

    model.compile(optimizer=optimizer, loss=loss_function, metrics={'labels': metrics,
                                                                    'autogen': tf.keras.metrics.CosineSimilarity()})

    model.summary()
    return model


# Variables and Everything
root_dir = '../../datasets/unzipped/'
dataset = 'InlineSkate'
validation_split = 0.2
num_rocket_kernels = 10000
epochs = 200
batch_size = 16

# Defining the Learning Rate
def learning_rate_schedule(epoch, lr):
    if epoch < 40:
        return 0.001
    elif 40 < epoch < 80:
        return 0.0001
    elif 80 < epoch < 150:
        return 0.00001
    else:
        return 1e-6


callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5),
             tf.keras.callbacks.LearningRateScheduler(learning_rate_schedule)]

# Load Training Dataset
mode = 1
train_x, train_y = load_single_dataset(root_dir+dataset+'/'+dataset+'_TRAIN.arff', seed=6343, mode=mode)
# Load Testing Dataset
test_x, test_y = load_single_dataset(root_dir+dataset+'/'+dataset+'_TEST.arff', seed=2345, mode=mode)
# Split Test Set into Validation and Testing
validation_size = int(test_x.shape[0] * validation_split)
valid_x, valid_y = test_x[:validation_size, :], test_y[:validation_size, :]
test_x, test_y = test_x[validation_size:, :], test_y[validation_size:, :]

# Extract Important Features about Dataset from Training Data
categories = len(np.unique(train_y, axis=0))
time_length = train_x.shape[1]

# Managing Dataset Imbalance
class_count = np.sum(train_y, axis=0)
class_weights = []
for i in range(categories):
    weight = 1 / class_count[i] * np.sum(class_count) / categories
    class_weights.append((i, weight))
class_weights = dict(class_weights)

# Initialize Model
model1 = dense_net(num_rocket_kernels, categories)

# Generate the Rocket Kernels
kernels = generate_kernels(time_length, num_rocket_kernels)

# Apply Kernels And Normalize - TrainX, TestX and ValidationX
new_train_x = apply_kernels(train_x, kernels)

new_test_x = apply_kernels(test_x, kernels)

new_val_x = apply_kernels(valid_x, kernels)

# Fit Model on New Feature Vector and Labels
"""h = model1.fit(x=new_train_x, y={'labels': train_y, 'autogen': new_train_x},
               validation_data=(new_val_x, {'labels': valid_y, 'autogen': new_val_x}),
               callbacks=callbacks, epochs=epochs, batch_size=batch_size)"""  # class_weight=class_weights)

h = model1.fit(x=new_train_x, y=train_y, validation_data=(new_val_x, valid_y),
               callbacks=callbacks, epochs=epochs, batch_size=batch_size, class_weight=class_weights)
# Save and View Performance
np.save('train_perf_'+dataset+'_'+str(epochs)+'_dense_net', list(h.history.values()))
# show_performance(list(h.history.values()), epochs, dataset)
eval_values = model1.evaluate(x=new_test_x, y=test_y)
np.save('eval_perf_'+dataset+'_'+str(epochs)+'_dense_net', eval_values)


