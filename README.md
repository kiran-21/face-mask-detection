# Face Mask Detection

Project Overview
This project is a Face Mask Detection System that detects whether a person is wearing a face mask or not using deep learning and image processing techniques. The model is built using Convolutional Neural Networks (CNNs) and is trained on an image dataset to classify people into two categories: with a mask and without a mask.

Project Objectives
- To detect whether a person is wearing a face mask to help reduce the spread of COVID-19.
- To implement feature selection methods for improved accuracy.
- To build a real-time face mask detection system using machine learning and deep learning techniques.

Technologies Used
- Python
- TensorFlow & Keras (Deep Learning Model)
- OpenCV (Image Processing)
- Flask (Web Deployment)
- Scikit-learn (Machine Learning)

System Requirements
Hardware Requirements
- RAM: 8GB
- Processor: Intel i5 / AMD Ryzen 5 or higher
- Storage: 1TB HDD or SSD
- Web Camera: Required for real-time detection

Software Requirements
- Anaconda (for package management)
- Python 3.10
- TensorFlow 2.x
- Keras
- OpenCV
- Flask

Installation Guide
Step 1: Clone the Repository
git clone https://github.com/kiran-21/face-mask-detection.git
cd face-mask-detection

Step 2: Create and Activate a Virtual Environment
conda create --name mask-detection python=3.10 -y
conda activate mask-detection

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Run the Model
python train.py  # Train the model (if not already trained)
python detect_mask_video.py  # Run real-time face mask detection

Step 5: Run the Web App (Flask Deployment)
python app.py

The web application will run at `http://127.0.0.1:5000/`.

Project Methodology
1. Data Collection: Images are collected from Kaggle and preprocessed.
2. Deep Learning: A CNN-based MobileNetV2 model is used for training.
3. Transfer Learning: We use a pre-trained MobileNetV2 model and fine-tune it.
4. Model Training:
   - Epochs: 20
   - Learning Rate: 0.0001
   - Batch Size: 100
5. Testing and Validation:
   - Accuracy achieved: 96%
   - Metrics: F1 Score, Precision, Recall
6. Deployment:
   - Flask web app for user-friendly interaction.
   - OpenCV for real-time video feed.

Model Performance
- Accuracy: 96%
- Loss Function: Sparse Categorical Cross-Entropy
- Optimizer: Adam
- Evaluation Metrics: F1 Score, Precision, Recall

Screenshots


Features
- Real-time face mask detection using webcam
- High accuracy with CNN-based deep learning model
- Web-based interface using Flask

Future Enhancements
- Improve accuracy with more training data.
- Deploy the model on cloud-based platforms.
- Implement face mask detection in low-light conditions.
- Add alerts when someone is detected without a mask.

References
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
- Flask Documentation: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)


