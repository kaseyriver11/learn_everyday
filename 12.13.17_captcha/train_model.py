import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
import imutils


def resize_to_fit(image_x, width, height):
    """
    A helper function to resize an image_x to fit within a given size
    :param image_x: image_x to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image_x
    """

    # grab the dimensions of the image_x, then initialize
    # the padding values
    (h, w) = image_x.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image_x = imutils.resize(image_x, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image_x = imutils.resize(image_x, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    pad_width = int((width - image_x.shape[1]) / 2.0)
    pad_height = int((height - image_x.shape[0]) / 2.0)

    # pad the image_x then apply one more resizing to handle any
    # rounding issues
    image_x = cv2.copyMakeBorder(image_x, pad_height, pad_height, pad_width, pad_width,
                                 cv2.BORDER_REPLICATE)
    image_x = cv2.resize(image_x, (width, height))

    # return the pre-processed image_x
    return image_x


LETTER_IMAGES_FOLDER = "data\solving_captchas\extracted_letter_images"
MODEL_FILENAME = "data\solving_captchas\captcha_model.hdf5"
MODEL_LABELS_FILENAME = "data\solving_captchas\model_labels.dat"


# initialize the data and labels
data = []
labels = []

# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the letter so it fits in a 20x20 pixel box
    image = resize_to_fit(image, 20, 20)

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)

    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-2]

    # Add the letter image and it's label to our training data
    data.append(image)
    labels.append(label)


# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=1111)
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(32, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network
model.fit(X_train, Y_train,
          validation_data=(X_test, Y_test), batch_size=32, epochs=1, verbose=1)

# Save the trained model to disk
model.save(MODEL_FILENAME)
