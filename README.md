# Image-Classification---Intel-Dataset

CNN Model Analysis Report
1. Introduction
This report provides an overview of the Convolutional Neural Network (CNN) implemented for image classification. The model was trained on the Intel Image Classification dataset, which consists of six different land categories. The goal was to build a deep learning model to accurately classify images into these categories.
________________________________________

2. Data Loading & Preprocessing
•	The dataset was loaded from the specified directories using ImageDataGenerator.
•	Training Data: 11,230 images belonging to 6 classes.
•	Testing Data: 2,804 images belonging to 6 classes.
•	Validation Data: A separate set was created using data augmentation techniques.
•	Image resizing and normalization were applied for better model performance.
________________________________________

3. Model Architecture
The CNN model was implemented using TensorFlow/Keras and consists of the following layers:
1.	Convolutional Layers: Feature extraction using multiple Conv2D layers with ReLU activation.
2.	Pooling Layers: MaxPooling2D layers to downsample feature maps and reduce computational complexity.
3.	Flatten Layer: Converts feature maps into a 1D array.
4.	Fully Connected Layers: Dense layers with dropout for regularization.
5.	Output Layer: A softmax activation function for multi-class classification.

Optimizer & Loss Function:
•	Optimizer: Adam (Adaptive Moment Estimation)
•	Loss Function: Categorical Crossentropy

Regularization Techniques:
•	Dropout layers to reduce overfitting.
•	Early stopping to halt training when validation loss stops improving.
________________________________________

4. Model Training & Evaluation
•	The model was trained on the dataset using a batch size of 32 and a certain number of epochs (not fully extracted).
•	Training accuracy and loss trends were monitored.
•	Model evaluation was performed on the test dataset.
 
________________________________________
5. Findings & Observations
•	The CNN successfully classified images into six categories.
•	The overall accuracy of the model is 84%.
•	The forest category had the highest classification performance (precision: 0.98, recall: 0.96, f1-score: 0.97), while buildings had the lowest (precision: 0.73, recall: 0.89, f1-score: 0.80).
•	Some misclassifications were observed, especially between glacier and mountain categories.
•	Training data augmentation helped improve generalization.
•	Early stopping ensured the model did not overfit.
________________________________________
6. Conclusion:
•	The CNN model performed well in classifying images from the dataset. The accuracy is 0.84 i.e., 84%.
•	Overall, the model provides a strong baseline for image classification and can be further optimized for better accuracy.

