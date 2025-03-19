import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import resample

# Variable to control oversampling
ENABLE_OVERSAMPLING = True  # Set to False to disable oversampling

def load_balanced_images_and_labels(directory, target_size=(96, 96)):
    images = []
    labels = []
    class_counts = {}

    # Load all images and labels from the directory
    for label_folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, label_folder)):
            label = int(label_folder)
            class_images = []
            for image_file in os.listdir(os.path.join(directory, label_folder)):
                image_path = os.path.join(directory, label_folder, image_file)
                image = load_img(image_path, target_size=target_size)
                image = img_to_array(image)
                class_images.append(image)
                labels.append(label)
            images.extend(class_images)
            class_counts[label] = len(class_images)

    # Perform oversampling only if ENABLE_OVERSAMPLING is True
    if ENABLE_OVERSAMPLING:
        # Find the class with the most images
        max_samples = max(class_counts.values())

        # Oversample for each class
        balanced_images = []
        balanced_labels = []
        for label_folder in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, label_folder)):
                label = int(label_folder)
                label_images = [img for img, lbl in zip(images, labels) if lbl == label]

                # Oversample images for this class
                resampled_images = resample(label_images, replace=True, n_samples=max_samples, random_state=123)
                balanced_images.extend(resampled_images)
                balanced_labels.extend([label] * max_samples)

        return np.array(balanced_images), np.array(balanced_labels)
    else:
        return np.array(images), np.array(labels)

# Define the CNN models
def model_cnn1(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def model_cnn2(input_shape, num_classes):
    model = Sequential([
        Conv2D(16, (5, 5), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(32, (5, 5), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (5, 5), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Load data
X_train, y_train = load_balanced_images_and_labels('train')
X_test, y_test = load_balanced_images_and_labels('test')

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Parameters for grid search
batch_sizes = [32, 64, 128]
learning_rates = [0.001, 0.0005, 0.0001]
epochs_list = [10, 20, 30]

# Function to train and save the model
def train_and_save_model(model, lr, batch_size, epochs, model_name):
    # Create directory based on oversampling status
    folder_name = "models_Oversampling_ON" if ENABLE_OVERSAMPLING else "models_Oversampling_OFF"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Compile, train, and save the model
    model.compile(optimizer=Adam(learning_rate=lr), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    model.save(os.path.join(folder_name, f'{model_name}_lr{lr}_batch{batch_size}_epochs{epochs}.h5'))
    return history

# Grid search over parameters
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for epochs in epochs_list:
            print(f"Training with batch size {batch_size}, learning rate {learning_rate}, epochs {epochs}")
            # Model 1
            model1 = model_cnn1((96, 96, 3), 5)
            history_cnn1 = train_and_save_model(model1, learning_rate, batch_size, epochs, "model_cnn1")

            # Model 2
            model2 = model_cnn2((96, 96, 3), 5)
            history_cnn2 = train_and_save_model(model2, learning_rate, batch_size, epochs, "model_cnn2")
