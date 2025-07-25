# Online Transaction Fraud Detection

![Project Banner](https://via.placeholder.com/800x200?text=Online+Transaction+Fraud+Detection)

## 📌 Overview
This project focuses on detecting fraudulent online transactions using a hybrid AI approach combining **Machine Learning (Random Forest)** and **Deep Learning (ANN)**. The system addresses the challenge of extreme class imbalance (fraudulent transactions <1%) and achieves high recall (100%) and precision (75%) through ensemble modeling.

---

## 🚀 Key Features
- **Hybrid AI Model**: Combines ANN (high recall) and Random Forest (high precision) with Logistic Regression as a meta-classifier.
- **Data Balancing**: Uses **SMOTE** and **Random Under Sampling** to handle imbalanced datasets.
- **Web Application**: Built with **Streamlit** for EDA, model insights, and real-time fraud prediction.
- **Performance Metrics**:  
  - Recall: 100%  
  - Precision: 75%  
  - F1-Score: 93-95%  
  - Accuracy: 89%  

---

## 🛠️ Technologies Used
- **Programming Language**: Python 3.10+  
- **Frameworks & Libraries**:  
  - Data Handling: `Pandas`, `NumPy`  
  - Visualization: `Matplotlib`, `Seaborn`, `Plotly`  
  - ML/DL: `Scikit-learn`, `TensorFlow/Keras`, `Imbalanced-learn`  
  - Deployment: `Streamlit`  
- **Hardware**: GPU (8GB+ VRAM recommended for training).  

---

## 📊 Methodology
1. **Data Preprocessing**:  
   - Applied SMOTE and Random Under Sampling to balance classes.  
2. **Model Development**:  
   - ANN for recall optimization.  
   - Random Forest for precision.  
   - Stacked ensemble with Logistic Regression.
   - Thus creating a Meta Model
3. **Evaluation**:  
   - Metrics: Accuracy, Precision, Recall, F1-Score.  
4. **Deployment**:  
   - Streamlit web app with EDA dashboard and prediction interface.  

---

## 🖥️ Web App Screenshots
| Home Page | Prediction Interface | EDA Dashboard |
|-----------|----------------------|---------------|
| ![Home](https://via.placeholder.com/300x200?text=Home+Page) | ![Prediction](https://via.placeholder.com/300x200?text=Prediction) | ![EDA](https://via.placeholder.com/300x200?text=EDA+Dashboard) |

---

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/online-fraud-detection.git
