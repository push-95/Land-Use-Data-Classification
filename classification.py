# All import dependencies can be installed from requirements.tx

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Load the train and test data

X = np.load('data/x_train.npy')
y = np.load('data/y_train.npy')
X_test = np.load('data/x_val.npy')
num_classes = 6



# Sample plot of random training image

#plt.imshow(X[11111,:,:,0:3])
#plt.show()



#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.16, random_state=42)

X_train = X
y_train = y



# Create model architecture with four layers using Keras

def create_model():
    # Initialize sequential model
    classifier = Sequential()

    # Layer 1
    classifier.add(Conv2D(32, (3, 3), input_shape = X_train.shape[1:], activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Layer 2
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Layer 3
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Ouput Layer
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(num_classes, activation = 'softmax'))
    
    return classifier


# Supervised training of the model using the above architeccture and training dataset (X_train, y_train)

def train_model():
    classifier = create_model()
    
    # Compile model
    classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    # Use a split of 67% training and 33% validation 
    classifier.fit(x=X_train, y=y_train, validation_split=0.33, batch_size=50, epochs=18, verbose=1)
    
    # Save weights for future use
    classifier.save_weights('models/ninth_model.h5')
    
    return classifier
    # eighth_model is the current best




# To train new model
# Compiling is taken care of in the train_model() function

#model = train_model()




# To load saved weights of a previous model

# Initialize model architecture
model_loaded = create_model()
# Load saved weights
model_loaded.load_weights('models/eighth_model.h5')
# Compile model
model_loaded.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])



# Save predicted values in csv file

classes = model_loaded.predict_classes(X_test, batch_size=32)
np.savetxt('y_predicted_val.csv', (classes))