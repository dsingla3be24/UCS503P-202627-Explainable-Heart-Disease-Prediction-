# 🚀 Week 4: Backend Machine Learning Pipeline & CNN Baseline

**📅 Date:** September 4 – September 10, 2026  
**🎯 Core Task:** Architected and developed the foundational Convolutional Neural Network (CNN) pipeline, including data ingestion, standardization, and Gramian Angular Summation Field (GASF) transformations.

---

### 🛠️ Key Contributions & Decisions

* **Environment Setup:** Configured the local Python environment with the required deep learning and data science libraries, including `tensorflow/keras`, `scikit-learn`, `pandas`, and `pyts`.
* **Data Ingestion & Preprocessing:** Successfully loaded the UCI Cleveland Heart Disease dataset, handled missing values, and standardized the 13 physiological features using `MinMaxScaler` to ensure numerical stability during neural network training.
* **GASF Image Transformation:** Implemented the mathematical pipeline to convert the 1D standardized clinical arrays into 2D polar coordinate image matrices. This step is critical because it allows the CNN to extract spatial relationships between different clinical features.
* **CNN Baseline Construction:** Architected the baseline Sequential CNN model using Keras. Designed the `Conv2D` and `MaxPooling2D` layers to effectively parse the GASF images, ending with a fully connected `Dense` layer utilizing a Sigmoid activation function to output the final heart disease probability.
* **Initial Validation:** Verified that data flows correctly from the CSV source, through the GASF transformer, and into the CNN architecture without dimensionality mismatch errors.

### 📎 Attachments
* [CardioAI_CNN_Baseline.ipynb](https://github.com/dsingla3be24/UCS503P-202627-Explainable-Heart-Disease-Prediction-/tree/master/code/backend/CardioAI_CNN_Baseline.ipynb)
