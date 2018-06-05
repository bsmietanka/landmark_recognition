from os.path import join, exists
from os import listdir, environ, makedirs
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential, Model, load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Input, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import VGG16, DenseNet121
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import argparse
from scipy import misc
import numpy as np
from skimage import transform
from matplotlib import pyplot as plt
from datetime import datetime

class model:

    types = {'VGG16', 'DenseNet121', 'VGG-based'}

    '''
    This function accepts both path to a dataset directory
    and to a .csv file containing classes and urls to photos.
    Function expects dataset directory to consist of two
    subdirectories: 'train' and 'validation'
    '''
    def prepare_data_generators(self, path, only_test=False):
        self.batch_size = 8

        if not only_test:
            train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

            self.train_generator = train_datagen.flow_from_directory(
                join(path, "train"),
                target_size=(200, 200),
                batch_size=self.batch_size,
                class_mode='categorical')

        test_datagen = ImageDataGenerator(rescale=1./255)

        self.validation_generator = test_datagen.flow_from_directory(
                join(path, 'validation'),
                target_size=(200, 200),
                batch_size=self.batch_size,
                class_mode='categorical')

        self.num_classes = len(self.validation_generator.class_indices)

        self.classes_dict = {}
        for i, dir in enumerate(listdir(join(path, "train"))):
            self.classes_dict[i] = dir

    def instantiate_model(self, model_type, freeze_conv=False):

        self.type = model_type
        eps = 1.1e-5

        if self.type == 'DenseNet121':
            pretrained_conv_model = DenseNet121(weights="imagenet", include_top=False)
        
            if freeze_conv:
                for layer in pretrained_conv_model.layers:
                    layer.trainable = False

            input = Input(shape=(None, None, 3),name = 'image_input')
            output_pretrained_conv = pretrained_conv_model(input)

            final_stage = "final"
            x = BatchNormalization(epsilon=eps, axis=3, name='conv'+str(final_stage)+'_blk_bn')(output_pretrained_conv)
            x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
            x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
            x = Dense(self.num_classes, name='predictions')(x)
            x = Activation('softmax', name='prob')(x)

            self.model = Model(input=input, output=x)

        elif self.type == 'VGG16':
            pretrained_conv_model = VGG16(weights='imagenet', include_top=False)

            if freeze_conv:
                for layer in pretrained_conv_model.layers:
                    layer.trainable = False

            input = Input(shape=(None, None, 3),name = 'image_input')
            output_pretrained_conv = pretrained_conv_model(input)

            x = BatchNormalization(epsilon=eps, axis=3, name='batch_normalization')(output_pretrained_conv)
            x = Activation('relu', name='relu_act')(x)
            x = GlobalAveragePooling2D()(output_pretrained_conv)
            x = Dense(512, activation='relu', name='fc1')(x)
            x = Dense(256, activation='relu', name='fc2')(x)
            x = Dense(self.num_classes, activation="softmax")(x)

            self.model = Model(input=input, output=x)

        elif self.type == 'VGG-based':
            self.model = Sequential()

            self.model.add(Convolution2D(64, (3,3), activation="relu", input_shape=(None, None, 3)))
            self.model.add(Convolution2D(64, (3,3), activation="relu", padding="same"))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
            self.model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
            self.model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
            self.model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
            self.model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(BatchNormalization(epsilon=eps, axis=3, name='batch_normalization'))
            self.model.add(Activation('relu', name='relu_act'))
            self.model.add(GlobalMaxPooling2D())
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(self.num_classes, activation='softmax'))
        
        self.model.summary()
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    # TODO: validate loaded model?
    # TODO: load classes dict?
    def load_model(self, path):
        self.type = "loaded"
        self.model = load_model(path)
        self.model.summary()

    def validate(self):
        return self.model.evaluate_generator(self.validation_generator)

    def train(self):
        output_dir = join("results", "{}_{}_{:%Y.%m.%d__%H-%M}".format(self.type, self.num_classes, datetime.now()))
        if not exists(output_dir):
            makedirs(output_dir)

        checkpoint = ModelCheckpoint(join(output_dir, 'weights.hdf5'), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')
        callbacks_list = [checkpoint, stopping]

        history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=50,
            validation_data=self.validation_generator,
            validation_steps=len(self.validation_generator),
            callbacks=callbacks_list)

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(join(output_dir, 'train_history_accuracy.pdf'))
        plt.savefig(join(output_dir, 'train_history_accuracy.png'))
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(join(output_dir, 'train_history_loss.pdf'))
        plt.savefig(join(output_dir, 'train_history_loss.png'))
        plt.close()

    # TODO: path to directory and single image?
    def predict(self, path):
        # classes_dict = {}
        # for i, dir in enumerate(listdir(join("dataset_expanded", "train"))):
        #     classes_dict[i] = dir
        photo = misc.imread(path)
        photo = transform.resize(photo, (200,200))
        prediction = self.model.predict(photo[np.newaxis])
        classes = prediction.argmax(axis=-1)
        print(f"Prediction for {path} :\n{classes}")
        # print(f"Class: {classes_dict[np.argmax(prediction)]}")
