# Homework 1: 10-class Classification in Machine Learning 🚀

**Course:** Robotics and Artificial Intelligence  
**Master:** Sapienza University of Rome  
**Student:** Edoardo Caciolo (Matricola: 1793918)  

---

## Introduction

This project aims to solve a multiclass classification problem (10 classes) on two different datasets, one with 100 features and the other with 1000 features.  
The goal is to analyze and compare various Machine Learning models, evaluating their performance in terms of accuracy and computational efficiency. 🔍

## Models Used

- **Logistic Regression** 📈
- **Decision Tree** 🌳
- **Random Forest** 🌲
- **Support Vector Machine (SVM)** 🔒
- **K-Nearest Neighbors (KNN)** 🤝
- **XGBoost** ⚡

_The analysis was performed using both with and without Cross Validation (CV, k=5)._

## Methodology

- **Preprocessing:** Data standardization using `StandardScaler`.
- **Optimization:** Hyperparameter tuning with `GridSearchCV` (except for Logistic Regression due to high computation time).
- **Data Splitting:** The dataset was divided into training and evaluation sets in a 5:1 ratio.
- **Evaluation:** Metrics such as Accuracy, Precision, Recall, and F1 Score were computed, along with an analysis of computational resources (execution time, CPU, RAM). ⏱️💻

## Results

- **Dataset 1 (100 features):**
  - The SVM model without CV showed excellent performance (accuracy ≃ 98.69%) with reasonable computation times.
  - Using CV slightly improved accuracy (accuracy ≃ 98.86%) but increased resource usage.
  
- **Dataset 2 (1000 features):**
  - Overall performance is slightly lower compared to Dataset 1.
  - The Random Forest without CV was chosen as the best compromise between accuracy and execution time.

## Conclusions

The analysis demonstrated that optimal model selection must balance accuracy, computation time, and resource usage.  
The comparison shows that there is no one-size-fits-all solution; the final choice was based on a compromise considering the specific characteristics of each dataset.  
For Dataset 1, SVM without CV was selected, while for Dataset 2, Random Forest without CV was chosen. ✅

## Attached Files

- **Report:** [ML_Homework1_Report_1793918.pdf](./ML_Homework1_Report_1793918.pdf) 📄
- **Source Code:** [ML_Homework1_1793918.zip](./ML_Homework1_1793918.zip) 📦
- **Predictions:**
  - Dataset 1: [d1_1793918.csv](./d1_1793918.csv)
  - Dataset 2: [d2_1793918.csv](./d2_1793918.csv)

---

## Appendix

The repository also includes the complete code used for the analysis, which comprises:

- `load_data.py`: Functions for loading data.
- `ML_Homework1_SourceCode_1793918.ipynb`: Notebook with the implementation and analysis of the models.

---

*This project was completed as part of Homework 1 for the Machine Learning course in the Robotics and Artificial Intelligence Master program at Sapienza University of Rome.*



---
---


# Homework 2: CNN-based Control for Virtual Car Racing 🚗🏁

**Course:** Machine Learning  
**Master:** Robotics and Artificial Intelligence, Sapienza University of Rome  
**Student:** Edoardo Caciolo (Matricola: 1793918)  

---

## Introduction

This project addresses the control problem for a racing car in a virtual Gymnasium environment using Convolutional Neural Networks (CNNs). The objective is to design, train, and evaluate CNN models that enable the car to navigate a track by processing pre-classified images and mapping them to discrete driving actions. 🎯

## Project Scope

- **Control Task:** Develop a control system for a virtual racing car with discrete actions (e.g., steer left/right, accelerate, brake).
- **Data:** Train the models on a dataset of 6,369 pre-classified images, each labeled with a specific driving action.
- **Models:** Two CNN architectures (CNN1 and CNN2) were implemented and compared.
- **Evaluation:** Models are evaluated using classic metrics (Accuracy, Precision, Recall, F1 Score, Average Evaluation Metrics) and the Total Reward (TR) obtained during simulation.

## Models Used

- **CNN1 Model** 🧠  
  - Features multiple convolutional layers followed by pooling, a flatten layer, dense layers, and dropout.
- **CNN2 Model** 🧠  
  - Incorporates larger filter sizes and Batch Normalization to stabilize and accelerate training.

Both models were trained under various hyperparameter settings using Grid Search, testing parameters such as Batch Size, Learning Rate, and the number of Epochs.

## Methodology

- **Preprocessing:**  
  - Image pixel values are normalized (dividing by 255.0) to scale them from 0 to 1.
  
- **Hyperparameter Tuning:**  
  - Grid Search was used to explore different combinations of Batch Size, Learning Rate, and Epochs.
  
- **Oversampling Analysis:**  
  - The dataset was initially imbalanced. Experiments with oversampling were conducted; however, the final model was trained on unbalanced data to better reflect real-world conditions.
  
- **Evaluation Metrics:**  
  - Accuracy, Precision, Recall, F1 Score, and Average Evaluation Metrics (AEM) were computed.
  - **Total Reward (TR)** was also recorded during simulation to assess real-time performance on the virtual track.

## Results & Model Choice

- **Training Outcomes:**  
  - Various models were generated, and performance was compared using evaluation metrics and simulation results.
  - The final model selected is **m 200013210.h5**, which demonstrated excellent simulation performance, achieving very good speed control, cornering grip, and trajectory recovery. ✅

- **Simulation Observations:**  
  - The chosen model maintained steady control, with only minor deviations that were quickly corrected.
  - Comparative tests highlighted that oversampling degraded learning in the simulation environment, confirming the decision to use the unbalanced training data.

## Future Developments

Potential future improvements include:
- **Integrating Deep Q-Networks (DQN)** to enhance decision-making in dynamic environments.
- **Combining with Model Predictive Control (MPC)** for optimized trajectory planning.
- A **hybrid approach** that leverages the strengths of CNNs for image processing, DQNs for strategic decision-making, and MPC for precise control actions could further enhance performance in both virtual and physical racing scenarios.

## Attached Files

- **Report:** [ML_Homework2_Report_1793918.pdf](./ML_Homework2_Report_1793918.pdf) 📄
- **Source Code Files:**  
  - `ML_Homework2_SourceCode_1793918.py` (CNN models generation)  
  - `ML_Homework2_evaluation_metrics_1793918.py` (Evaluation of the models)  
  - `ML_Homework2_play_policy_template_1793918.py` (Simulation code for the virtual environment)  
- **Model Folders:**  
  - *models Oversampling OFF* (39 CNN models)  
  - *models Oversampling ON* (54 CNN models)  
- **Evaluation Spreadsheets:** MER ML Homework 2 1793918.xlsx (contains detailed metrics and confusion matrices) 📊
- **Simulation Video:** ML Homework 2 1793918 m 200013210 simulation video.mp4 🎞️
- **GitHub Repository:**  
  [Project 2 Machine Learning GitHub Repository](https://github.com/Ed0C97/Project-2-Machine-Learning.git)

---

## Appendix

The repository also contains the complete source code used for model training, evaluation, and simulation, including:
- `load_img` and image processing scripts.
- CNN model definitions (see Appendix A in the report for details).
- Detailed evaluation tables and simulation results.

---

*This project was developed as part of Homework 2 for the Machine Learning course in the Robotics and Artificial Intelligence Master's program at Sapienza University of Rome. Enjoy the ride! 🚀*



