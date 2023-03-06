# Import required libraries
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import requests
from io import BytesIO
from PIL import Image

# Define function to download image from URL and convert it to a NumPy array
def download_and_convert_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((32, 32))
    img_array = np.array(img)
    return img_array

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocess the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the neural network architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define function to continuously read live data and make predictions
def predict_from_live_data():
    while True:
        url = 'https://picsum.photos/200'
        img_array = download_and_convert_image(url)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        print(prediction)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Start making predictions from live data
predict_from_live_data()
