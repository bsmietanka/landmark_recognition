from os.path import join
from os import listdir, environ
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

# TODO: think of sensible architecture for each pretrained network
# TODO: add non-landmark class
# TODO: expand dataset
# TODO: compare three architectures (in cocumentation)
# TODO: more accurate training results (eg. accuracy in epochs)
# TODO: working validation

def modified_pretrained_model(classes, pretrained_weights="DenseNet121", freeze_top=False):

    if pretrained_weights == "DenseNet121":
        pretrained_conv_model = DenseNet121(weights="imagenet", include_top=False)
        
        if freeze_top:
            for layer in pretrained_conv_model.layers:
                layer.trainable = False

        input = Input(shape=(None, None, 3),name = 'image_input')
        output_pretrained_conv = pretrained_conv_model(input)

        eps = 1.1e-5
        final_stage = "final"
        x = BatchNormalization(epsilon=eps, axis=3, name='conv'+str(final_stage)+'_blk_bn')(output_pretrained_conv)
        x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
        x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
        x = Dense(classes, name='predictions')(x)
        x = Activation('softmax', name='prob')(x)

    elif pretrained_weights == "VGG16":
        pretrained_conv_model = VGG16(weights='imagenet', include_top=False)

        if freeze_top:
            for layer in pretrained_conv_model.layers:
                layer.trainable = False

        input = Input(shape=(None, None, 3),name = 'image_input')
        output_pretrained_conv = pretrained_conv_model(input)

        eps = 1.1e-5
        x = BatchNormalization(epsilon=eps, axis=3, name='batch_normalization')(output_pretrained_conv)
        x = Activation('relu', name='relu_act')(x)
        x = GlobalAveragePooling2D()(output_pretrained_conv)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dense(classes, activation="softmax")(x)

    model = Model(input=input, output=x)
    model.summary()

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    model.save(f"pretrained-weights-{pretrained_weights}.hdf5")
    return model

def create_new_model(classes):
    model = Sequential()

    eps = 1.1e-5

    model.add(Convolution2D(64, (3,3), activation="relu", input_shape=(None, None, 3)))
    model.add(Convolution2D(64, (3,3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(epsilon=eps, axis=3, name='batch_normalization'))
    model.add(Activation('relu', name='relu_act'))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    model.summary()

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    return model

# TODO: experiment with batch size
# TODO: add custom generator with variable image size for each batch

def main():
    parser = argparse.ArgumentParser(description='Deep neural network model for landmark recognition')
    parser.add_argument('dataset_path', metavar='dataset_path', nargs='?', default='', help='path to a dataset directory')
    # parser.add_argument('-k', type=int, default=5, help='number of neighbours')
    # read dataset from csv and download to local directiory?
    parser.add_argument('-m', '--model',
                        default="DenseNet121",
                        choices=["DenseNet121", "VGG16", "custom"],
                        help='indicates which model should be used')
    parser.add_argument('-p', '--predict', help='path to photo to predict landmark')
    parser.add_argument('-l', '--load', help='path to saved model weights')
    parser.add_argument('-f', '--freeze', action='store_true', help='freeze top convolutional layers when using pretrained weights')
    parser.add_argument('-v', '--validate', action='store_true', help='validation directory')
    args = parser.parse_args()

    if args.predict is not None and args.load is None:
        print("Provide saved model path to predict landmark")
        return

    if args.predict is None:
        # TODO calculate batch size based on available memory
        batch_size = 8

        if not args.validate:
            train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1./255)

        if not args.validate:
            train_generator = train_datagen.flow_from_directory(
                    join(args.dataset_path, "train"),
                    target_size=(200, 200),
                    batch_size=batch_size,
                    class_mode='categorical')

            train_batches = len(train_generator)

        validation_generator = test_datagen.flow_from_directory(
                join(args.dataset_path, 'validation'),
                target_size=(200, 200),
                batch_size=batch_size,
                class_mode='categorical')

        num_classes = len(validation_generator.class_indices)
        test_batches = len(validation_generator)
        classes_dict = {}
        for i, dir in enumerate(listdir(join("dataset_expanded", "train"))):
            classes_dict[i] = dir

    if args.load is not None:
        # TODO validate model
        model = load_model(args.load)
    elif args.model == "custom":
        model = create_new_model(num_classes)
    else:
        model = modified_pretrained_model(
            num_classes,
            pretrained_weights=args.model,
            freeze_top=args.freeze)

    if args.predict is None:
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        callbacks_list = [checkpoint, stopping]

        if not args.validate:
            model.fit_generator(
                train_generator,
                steps_per_epoch=train_batches,
                epochs=50,
                validation_data=validation_generator,
                validation_steps=test_batches,
                callbacks=callbacks_list)
        else:
            score, acc = model.evaluate_generator(validation_generator, verbose=1)
            print(f"Model test score: {score}")
            print(f"Model test accuracy: {acc}")

    else:
        classes_dict = {}
        for i, dir in enumerate(listdir(join("dataset_expanded", "train"))):
            classes_dict[i] = dir
        photo = misc.imread(args.predict)
        photo = transform.resize(photo, (200,200))
        model.summary()
        prediction = model.predict(photo[np.newaxis])
        print(f"Prediction for {args.predict} : {prediction}")
        print(f"Class: {classes_dict[np.argmax(prediction)]}")

# TODO: argparse in __main__,
# rename main() to something more verbose,
# add argparse arguments
if __name__ == "__main__":
    main()
