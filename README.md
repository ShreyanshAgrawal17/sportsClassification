# 🏆 Sports Classification using CNN & DenseNet

This repository contains a deep learning project for **classifying sports images** into five categories:  
🏏 **Cricket** | ⚽ **Football** | 🏊 **Swimming** | 🏑 **Field Hockey** | 🏀 **Basketball**  

The project implements **CNN** and **DenseNet-121 (Transfer Learning)** to achieve high accuracy.

---

## 📌 Project Overview

This project aims to classify sports images using **deep learning models**.  
It uses **ImageDataGenerator** for data preprocessing and augmentation.  
Models are evaluated on a **separate test dataset**.

---

## 🚀 Models Implemented

### 1️⃣ **DenseNet-121 (Transfer Learning)**
✔ Uses **DenseNet-121** pre-trained on ImageNet.  
✔ Freezes base layers and adds a **custom classification head**.  
✔ **Adam optimizer, categorical cross-entropy loss, dropout (0.5)** to prevent overfitting.  
✔ **Early stopping & ModelCheckpoint** for best model selection.  

### 2️⃣ **Custom CNN Model**
✔ **3 Conv2D layers** with ReLU activation & MaxPooling.  
✔ Fully connected layers for classification.  
✔ **Dropout (0.5)** to reduce overfitting.  
✔ Uses **categorical cross-entropy loss & Adam optimizer**.  

---

## 🗂 Dataset

The dataset consists of images classified into **five sports categories**:  
- 🏏 **Cricket**  
- ⚽ **Football**  
- 🏊 **Swimming**  
- 🏑 **Field Hockey**  
- 🏀 **Basketball**  

Data preprocessing is done using **ImageDataGenerator** for:  
✅ **Rescaling**  
✅ **Data Augmentation**  
✅ **Train-validation split**  

---

## 📊 Evaluation & Results

### **DenseNet Model**
✅ Achieved **high accuracy** on test data.  
✅ Overfitting reduced with **dropout & early stopping**.  

### **CNN Model**
✅ Performed well but slightly lower accuracy than DenseNet.  

#### **Metrics Used**:
✔ **Accuracy**  
✔ **Confusion Matrix**  
✔ **Classification Report**  

---

## 🔧 Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/sports-classification.git
cd sports-classification
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Models
```bash
python train.py
```

### 4️⃣ Evaluate on Test Data
```bash
python evaluate.py
```

---

## 📷 Model Predictions

To test the model with your own images, run:

```bash
python predict.py --image path/to/your/image.jpg
```

---

## 🔥 Future Improvements

🚀 Add more diverse datasets for better generalization.  
🚀 Fine-tune DenseNet instead of freezing all layers.  
🚀 Experiment with ResNet, EfficientNet, or Vision Transformers.  

---

## 🤝 Contributing

Feel free to fork this repository.  

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

## ✨ Author

👨‍💻 **Shreyansh Agrawal**  
Passionate about deep learning & computer vision.
