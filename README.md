Sports Classification using CNN and DenseNet
This repository contains a deep learning project for classifying sports images into five categories: Cricket, Football, Swimming, Field Hockey, and Basketball. The models used include a Convolutional Neural Network (CNN) and a pre-trained DenseNet model for improved accuracy.

📌 Project Overview
The goal of this project is to classify images into one of the five sports categories using deep learning techniques. The dataset is processed using TensorFlow’s ImageDataGenerator, and models are evaluated on separate test data.

🚀 Models Implemented
1️⃣ DenseNet-121 (Transfer Learning)
Utilizes DenseNet-121 with pre-trained ImageNet weights.
Freezes DenseNet layers and adds a custom classification head.
Uses Adam optimizer, categorical cross-entropy loss, and dropout (0.5) to reduce overfitting.
Early stopping and model checkpointing ensure the best model is saved.
2️⃣ Custom CNN Model
Three Conv2D layers with ReLU activation and MaxPooling.
Fully connected layers for classification.
Uses dropout (0.5) to prevent overfitting.
Trained with categorical cross-entropy loss and Adam optimizer.
🗂 Dataset
The dataset consists of images categorized into five sports classes:
🏏 Cricket
⚽ Football
🏊 Swimming
🏑 Field Hockey
🏀 Basketball
Data is preprocessed using ImageDataGenerator, which applies rescaling, augmentation, and validation split.
📊 Evaluation & Results
DenseNet Model:

Achieved high accuracy on test data.
Overfitting was reduced using dropout and early stopping.
CNN Model:

Performed well but had slightly lower accuracy compared to DenseNet.
Metrics Used:

✅ Accuracy
✅ Confusion Matrix
✅ Classification Report
🔧 Setup & Installation
1️⃣ Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/sports-classification.git
cd sports-classification
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Train the Models
python
Copy
Edit
python train.py
4️⃣ Evaluate on Test Data
python
Copy
Edit
python evaluate.py
📷 Model Predictions
To test the model with your own images, run:

python
Copy
Edit
python predict.py --image path/to/your/image.jpg
🔥 Future Improvements
Add more diverse datasets to improve generalization.
Fine-tune DenseNet instead of freezing all layers.
Experiment with ResNet, EfficientNet, or Vision Transformers.
🤝 Contributing
Feel free to fork this repository, make improvements, and submit a pull request. Contributions are welcome! 😊

📜 License
This project is open-source and available under the MIT License.

✨ Author
Shreyansh
💻 Passionate about deep learning and computer vision.
