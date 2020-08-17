from datetime import datetime
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten
)
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import one_hot
import constants


def create_model(input_shape, classes):
    '''
    Creates a 2D Convolutional Neural Network

    Arguments:
    input_shape -- tensor of dimension (height, width, no_of_channels)
    classes -- scalar representing no. of classes

    Returns:
    model -- model of the keras Model class
    '''

    i = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(i)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(4, 4))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(8, 8))(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(i, x)
    return model


def compile_model(model,
                  optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']):
    '''
    Compiles the 2D CNN with loss, metrics and optimizer

    Arguments:
    optimizer -- keras optimizer
    loss -- loss funtion/objective function
    metrics -- list of metrics needed for output

    Returns: None
    '''

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def training_data_augmentation(x_train, y_train, batch_size):
    '''
    Creates a data generator to perform image data augmentation during run time

    Arguments:
    x_train -- tensor of dimension (batch_size, height, width, no_of_channels)
    y_train -- tensor of dimension (batch_size, no_of_classes)
    batch_size -- scalar representing the batch_size

    Returns:
    train_generator -- generator of the keras ImageDataGenerator class
    '''

    data_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    train_generator = data_generator.flow(
        x_train, y_train, batch_size=batch_size)
    return train_generator


def train_model(model, train_generator, x_test, y_test, no_of_epochs):
    '''
    Trains the model upto specified no. of epochs

    Arguments:
    model -- compiled keras model
    train_generator -- generator of the keras ImageDataGenerator class
    x_test -- tensor of dimension (batch_size, height, width, no_of_channels)
    y_test -- (batch_size, no_of_classes)
    no_of_epochs -- scalar representing no. of epochs

    Returns:
    r -- keras History object
    '''

    r = model.fit_generator(train_generator, validation_data=(
        x_test, y_test), epochs=no_of_epochs)
    return r


def evaluate_model(model, x_test, y_test, batch_size):
    '''
    Evaluates the trained model to obtain validation metrics

    Arguments:
    model -- trained keras model
    x_test -- tensor of dimension (batch_size, height, width, no_of_channels)
    y_test -- (batch_size, no_of_classes)
    batch_size -- scalar representing batch_size

    Returns:
    loss -- scalar representing validation loss
    accuracy -- scalar representing validation accuracy
    '''

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(loss, accuracy)


def save_model(model):
    '''
    Saves keras model along with weights in .h5 format
    with name as current timestamp

    Arguments:
    model -- keras model

    Returns: None
    '''

    now = datetime.now()
    model_name_suffix = now.strftime('%d-%m-%Y-%H:%M:%S')
    save_model(model, constants.SAVE_MODEL_PATH +
               '/model${}.h5'.format(model_name_suffix))


def load_saved_model(model_name):
    '''
    Loads saved keras model in .h5 format along with weights

    Arguments:
    model_name -- string representing the name of the model

    Returns:
    loaded_model -- keras model with weights
    '''

    loaded_model = load_model(
        constants.LOAD_MODEL_PATH + '/{}'.format(model_name))
    return loaded_model


def predict_image(model, image):
    '''
    Outputs model predictions on a tensor

    Arguments:
    model -- trained/loaded keras model
    image -- tensor of dimension (batch_size, height, width, no_of_channels)

    Returns:
    (result_label, score) -- scalar representing class number in range [0, no_of_classes-1]
                             and scalar representing accuracy for predicted class respectively
    '''

    predicted_labels = model.predict(image)
    result_label = predicted_labels.argmax(axis=1)[0]
    score = max(predicted_labels[0]) * 100
    return (float(result_label), float(score))
