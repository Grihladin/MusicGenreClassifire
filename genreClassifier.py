import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# путь к файлу json, в котором хранятся MFCC и метки жанров для каждого обработанного сегмента
DATA_PATH = "путь к json файлу с данными"

def load_data(data_path):
    """Загружаем training dataset из json file.

        :param data_path (str): путь к файлу json, содержащему данные
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # конвертируем листы в массивы numpy
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("ВСЁ ЗАГРУЗИЛОСЬ!")

    return  X, y


if __name__ == "__main__":

    # занружаем данные
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # строю топологию сети
    model = keras.Sequential([

        # input ayer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # первый dense layer
        keras.layers.Dense(512, activation='relu'),

        # втрой dense layer
        keras.layers.Dense(256, activation='relu'),

        # третий dense layer
        keras.layers.Dense(64, activation='relu'),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # компилирую модель
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # тренирую модель
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)