# Lenet model

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, AvgPool2D, Dense, Flatten, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
from keras.utils import to_categorical

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

inputs = Input(shape=x_train.shape[1:])

x = Conv2D(filters=6, kernel_size=(5,5), padding='same', activation='sigmoid')(inputs)
x = AvgPool2D(pool_size=(2, 2), strides=2)(x)
x = Dropout(rate=0.1)(x)

x = Conv2D(filters=16, kernel_size=(5,5), activation='sigmoid')(x)
x = AvgPool2D(pool_size=(2, 2), strides=2)(x)
x = Dropout(rate=0.1)(x)

x = Conv2D(filters=120, kernel_size=(5,5), activation='sigmoid')(x)
x = Dropout(rate=0.1)(x)

x = Flatten()(x)

x = Dense(84, activation='sigmoid')(x)
x = Dropout(rate=0.2)(x)

predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 20
batch_size = 30

datagen = ImageDataGenerator(
  rotation_range=10,
  zoom_range=0.1,
  width_shift_range=0.1,
  height_shift_range=0.1
)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                              validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0]//batch_size)

print( model.summary())
scores = model.evaluate(x_test, y_test, verbose=1)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print("CNN EFFICIENCY: %2.f%%" % (scores[1]*100))

plt.plot(history.history['acc'], label='training accuracy')
plt.plot(history.history['val_acc'], label='testing accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='testing loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

  
             
