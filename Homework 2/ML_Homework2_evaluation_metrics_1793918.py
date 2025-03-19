import os
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Variable to control the use of models trained with oversampling
ENABLE_TEST_OVERSAMPLING = True  # Set to False to use models trained without oversampling

# Define the paths to model folders
models_path_with_oversampling = 'models_Oversampling_ON'  # Update with the correct path for models with oversampling
models_path_without_oversampling = 'models_Oversampling_OFF'  # Update with the correct path for models without oversampling
test_data_path = 'test'  # Update with the path to the test data

# Select the appropriate model folder
models_path = models_path_with_oversampling if ENABLE_TEST_OVERSAMPLING else models_path_without_oversampling
model_files = [file for file in os.listdir(models_path) if file.endswith('.h5')]

# Define data generator for test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(96, 96),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)

# Function to evaluate the model
def evaluate_model(model, test_generator):
    test_generator.reset()
    pred = model.predict(test_generator, steps=len(test_generator), verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)

    labels = (test_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    y_true = test_generator.classes
    accuracy = accuracy_score(y_true, predicted_class_indices)
    precision = precision_score(y_true, predicted_class_indices, average='macro', zero_division=0)
    recall = recall_score(y_true, predicted_class_indices, average='macro', zero_division=0)
    f1 = f1_score(y_true, predicted_class_indices, average='macro', zero_division=0)

    return accuracy, precision, recall, f1

# Function to plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, model_name, folder_name):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(f'{folder_name}/{model_name}_confusion_matrix.png', format='png')
    plt.close()

results = {}

# Create folder name for confusion matrices based on oversampling status
confusion_matrix_folder = 'confusion_matrices_Oversampling_ON' if ENABLE_TEST_OVERSAMPLING else 'confusion_matrices_Oversampling_OFF'
folder_name = confusion_matrix_folder

# Loop through model files and evaluate each model
for model_file in model_files:
    print(f"Evaluating model {model_file}...")
    model = load_model(os.path.join(models_path, model_file))
    accuracy, precision, recall, f1 = evaluate_model(model, test_generator)
    results[model_file] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

    y_pred = np.argmax(model.predict(test_generator), axis=1)
    y_true = test_generator.classes  # Utilizza test_generator.classes per ottenere le etichette corrette
    class_labels = list(test_generator.class_indices.keys())
    plot_confusion_matrix(y_true, y_pred, class_labels, model_file, confusion_matrix_folder)

# Save evaluation results to an Excel file
results_df = pd.DataFrame(results).transpose()

# Save evaluation results to an Excel file with a different name depending on oversampling status
if ENABLE_TEST_OVERSAMPLING:
    excel_filename = 'model_evaluation_results_oversampling_ON.xlsx'
else:
    excel_filename = 'model_evaluation_results_oversampling_OFF.xlsx'

results_df.to_excel(excel_filename)

# Display evaluation results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
    print("-----------")

