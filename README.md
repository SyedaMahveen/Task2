# Task2
!pip install tensorflow matplotlib

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

OUTPUT:
Collecting tensorflow
  Downloading tensorflow-2.17.0-cp310-cp310-manylinux_2_17_x86_64.whl (588.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 588.3/588.3 MB 10.2 MB/s eta 0:00:00
Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages
Successfully installed tensorflow-2.17.0

Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
11501568/11490434 [==============================] - 0s 0us/step

Epoch 1/5
1875/1875 [==============================] - 6s 2ms/step - loss: 0.2634 - accuracy: 0.9241 - val_loss: 0.1368 - val_accuracy: 0.9599
Epoch 2/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.1122 - accuracy: 0.9666 - val_loss: 0.0985 - val_accuracy: 0.9702
Epoch 3/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0775 - accuracy: 0.9762 - val_loss: 0.0820 - val_accuracy: 0.9744
Epoch 4/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0572 - accuracy: 0.9814 - val_loss: 0.0782 - val_accuracy: 0.9763
Epoch 5/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0438 - accuracy: 0.9860 - val_loss: 0.0749 - val_accuracy: 0.9771

313/313 - 0s - loss: 0.0749 - accuracy: 0.9771

Test accuracy: 97.71%