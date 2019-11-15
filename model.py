from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator
from keras.constraints import max_norm
from keras.optimizers import SGD

'''
For GPU support, tensorflow>=1.15.* must be installed, otherwise GPU packages can be installed with 
`pip install tensorflow-gpu==1.15`

This model builds on CUDA enabled NVIDIA GPUs with the following software requirements:
- NVIDIA® GPU drivers —CUDA 10.0 requires 410.x or higher.
- CUDA® Toolkit —TensorFlow supports CUDA 10.0 (TensorFlow >= 1.13.0)
    - CUPTI ships with the CUDA Toolkit.
- cuDNN SDK (>= 7.4.1)

CPU Training takes ~13 hours @ 50 epochs.
'''

chess_piece_types = ['bishop', 'rook', 'pawn', 'knight']

class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3), kernel_constraint=max_norm(2.),
                              bias_constraint=max_norm(2.)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3), kernel_constraint=max_norm(2.), bias_constraint=max_norm(2.)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), kernel_constraint=max_norm(2.), bias_constraint=max_norm(2.)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(chess_piece_types)))  # Equal to number of classes
        self.model.add(Activation('softmax'))

        # TODO: Evaluate other optimizers
        # opt = SGD(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def __save_weights(self, weights_file):
        self.model.save_weights(weights_file)

    def __load_weights(self, weights_file):
        self.model.load_weights(weights_file)

    def train(self, batch_size=16, epochs=10):
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            rotation_range=180,
            height_shift_range=0.1,
            width_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='categorical')

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=400,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=160)

        self.__save_weights('%s_epochs.h5' % epochs)

    def test(self, weights_file, img):
        self.__load_weights(weights_file)
