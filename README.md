# Hyperspectral Image Classification

Welcome to my repository! This project contains my machine learning pipeline developed for a university competition, where the goal was to classify complex hyperspectral image patches into specific thematic categories.

## The Challenge
Unlike standard RGB images, hyperspectral imaging captures a wide spectrum of light for each pixel, providing a massive amount of hidden detail. 
* **The Data:** The dataset contained 16,822 training patches. Each patch was a multi-dimensional tensor of shape `19 x 19 x 48` (representing spatial dimensions and 48 specific spectral bands in the 380–2500nm range).
* **The Objective:** Predict the correct thematic label (spanning from 1 to 7) for the central pixel of each patch in an unlabelled test set.
* **The Metric:** Models were strictly evaluated using the **Macro F1-Score**, which meant the solution had to perform exceptionally well across all classes, heavily penalizing models that ignored minority categories.

## My Approach and Architecture
Dealing with `19x19x48` tensors can quickly lead to overfitting if passed directly into basic models. I designed a custom pipeline focused on deep feature extraction, data normalization, and an ensembling strategy.

### 1. Data Preprocessing & Exploratory Data Analysis
Before feeding the data to any model, I analyzed the class distribution histograms. I discovered that classes **4** and **5** were virtually non-existent in the training data. To prevent the models from learning noise or becoming biased towards non-representative labels, I explicitly filtered out these classes from the training set, streamlining the classification task.

### 2. Feature Engineering
Instead of using raw spatial data, I extracted heavily targeted mathematical and statistical features to give the models a dense, highly informative vector:
* **Central Pixel Focus:** I isolated the spectral curve of the central pixel and calculated the 1st and 2nd-order discrete differences (gradients) to capture the shape and variations of the spectral signature.
* **Neighborhood Context:** I extracted the immediate 3x3 spatial neighborhood around the center and computed its mean and standard deviation across axes to provide spatial context.
* **Statistical Moments:** To summarize the spectral distribution, I calculated skewness, kurtosis, and the trapezoidal area under the spectral curve.

### 3. Advanced Data Transformation
To handle outliers and normalize the newly created feature space, I applied the **Yeo-Johnson Power Transformer**. This step was critical—it forced the feature distributions to become more Gaussian-like, which significantly improved the convergence rate and accuracy of the neural network.

### 4. Model Ensembling & Cross-Validation
I utilized a 10-fold Stratified Cross-Validation approach to train two vastly different model architectures and blended their predictions (55% / 45% split) to ensure maximum robustness:
* **LightGBM Classifier:** A gradient boosting model configured with balanced class weights, optimized leaf nodes, and GPU acceleration for fast training.
* **Multi-Layer Perceptron (MLP):** A deep neural network built with TensorFlow/Keras. I structured it with multiple Dense layers (512 and 256 neurons), using Batch Normalization for stability and Dropout layers (0.3) to heavily penalize overfitting. The learning rate was dynamically managed using `ReduceLROnPlateau` and `EarlyStopping` callbacks.

### 5. Pseudo-Labeling (Semi-Supervised Learning)
To squeeze out the absolute maximum performance on the test set, I implemented a safe pseudo-labeling strategy. I took the test predictions where the ensemble's confidence exceeded the 95% threshold, assigned them as "ground truth", and appended them back to the training dataset. Finally, I retrained a new LightGBM model from scratch on this augmented dataset to generate the final submission.

## Results and Impact
By combining rigorous feature engineering, intelligent data pruning, powerful ensembling, and semi-supervised pseudo-labeling, my model achieved a top **Macro F1-Score of 0.8528** on the private test data. 

This robust performance secured **2nd place** out of approximately 120 students participating in the competition.

## Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, LightGBM
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation:** NumPy, Pandas
