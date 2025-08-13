Vision AI in 5 Days: From Beginner to Image Recognition Expert

This project is my implementation from the **AI Image Recognition Bootcamp**, where I built and deployed multiple deep learning models for image classification — starting from scratch with a CNN to fine-tuning a pre-trained MobileNetV2 model.

Project Overview
Over 5 days, I learned and implemented:
1. **Image Preprocessing & EDA** – loading, scaling, reshaping, and visualizing datasets (MNIST, CIFAR-10, Cats vs. Dogs).
2. **Basic CNN Model** – building, training, and evaluating a custom convolutional neural network.
3. **Data Augmentation & Advanced Evaluation** – improving generalization and analyzing results with confusion matrices, classification reports, and ROC curves.
4. **Transfer Learning** – fine-tuning MobileNetV2 for Cats vs. Dogs classification and comparing performance with the custom CNN.
5. **Deployment & Predictions** – saving/loading trained models, predicting new data, and creating portfolio-ready visualizations.

Technologies Used
- Python 3
- TensorFlow / Keras
- scikit-learn
- Matplotlib & Seaborn
- Google Colab
- Pandas & NumPy
- Kaggle Datasets

---

## 📂 Project Structure
📁 Vision-AI-Bootcamp
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 Day1_Preprocessing_EDA.ipynb
├── 📄 Day2_CNN_Model.ipynb
├── 📄 Day3_Data_Augmentation_Evaluation.ipynb
├── 📄 Day4_Transfer_Learning.ipynb
├── 📄 Day5_Predictions_Deployment.ipynb
│
├── 📁 models
│   ├── cnn_model.h5
│   └── mobilenet_cats_dogs.h5
│
├── 📁 outputs
│   ├── accuracy_loss_curves.png
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_tl.png
│   ├── roc_curve.png
│   ├── sample_predictions.png
│
└── 📁 datasets

Datasets Used
- MNIST – Handwritten digits (10 classes)
- CIFAR-10 – Objects like airplanes, cars, dogs, cats
- Cats vs. Dogs – Binary classification dataset from Kaggle

---

How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/vision-ai-bootcamp.git
   cd vision-ai-bootcamp
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open notebooks in Google Colab or Jupyter Notebook.
4. Follow the steps in each day’s notebook to train, evaluate, and test models.

Model Performance

| Model                          | Accuracy | Precision | Recall | F1-Score |
|--------------------------------|----------|-----------|--------|----------|
| **Custom CNN (MNIST)**         | 0.98     | 0.98      | 0.98   | 0.98     |
| **Transfer Learning (Cats/Dogs)** | 0.84  | 0.84      | 0.84   | 0.84     |

Sample Predictions
| Image | Prediction |
|-------|------------|
| ![Dog](outputs/sample_dog.jpg) | Dog |
| ![Cat](outputs/sample_cat.jpg) | Cat |


LinkedIn Post Draft
> 🚀 Completed a 5-day AI Image Recognition Bootcamp where I built and deployed models using CNNs and Transfer Learning!  
> 🖼 Worked with datasets like MNIST, CIFAR-10, and Cats vs. Dogs.  
> 🔍 Achieved 98% accuracy on MNIST and 84% on Cats vs. Dogs using MobileNetV2.  
> 📊 Learned preprocessing, augmentation, evaluation metrics, and deployment.  

License
This project is for educational purposes as part of the AI Image Recognition Bootcamp.
