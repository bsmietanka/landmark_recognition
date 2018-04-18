from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Input, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16, DenseNet121

# TODO: tidy this function, 
# TODO: add parameter specifying which pretrained network should be used
# TODO: think of sensible architecture for each pretrained network
# TODO: load saved model if exists
# TODO: maybe experiment with optimizer

def modified_pretrained_model(classes):
    # pretrained_conv_model = VGG16(weights='imagenet', include_top=False)
    pretrained_conv_model = DenseNet121(weights="imagenet", include_top=False)
    # pretrained_conv_model.summary()

    #Create your own input format (here 3x200x200)
    input = Input(shape=(200, 200, 3),name = 'image_input')

    #Use the generated model 
    output_pretrained_conv = pretrained_conv_model(input)

    #Add the fully-connected layers 
#     x = Flatten(name='flatten')(output_pretrained_conv)
#     x = Dense(512, activation='relu', name='fc1')(x)
#     x = Dense(256, activation='relu', name='fc2')(x)
    eps = 1.1e-5
    final_stage = "final"
    x = BatchNormalization(epsilon=eps, axis=3, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=3, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    x = Dense(classes, name='predictions')(x)
    x = Activation('softmax', name='prob')(x)

    #Create your own model 
    model = Model(input=input, output=x)

    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    model.summary()

    model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    model.save("pretrained-weights.hdf5")

    return model

# TODO: load saved model if exists
# TODO: maybe experiment with optimizer

def create_new_model(classes):
    model = Sequential()

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
    model.add(GlobalMaxPooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

    model.summary()

    # load model if exists
    # model.load_weights("weights.best.hdf5")

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

# TODO: tidy up main function
# TODO: add argparse and cmdline parameters (eg. dataset path, model type etc.)
# TODO: experiment with learning rate, batch size

if __name__ == "__main__":
    # get num classes from dataset
    num_classes = 31
    # model = create_new_model(num_classes)
    model = modified_pretrained_model(num_classes)

    batch_size = 2

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            'dataset',  # this is the target directory
            target_size=(200, 200),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            'validation',
            target_size=(200, 200),
            batch_size=batch_size,
            class_mode='categorical')

    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # change steps per epoch
    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=callbacks_list)
