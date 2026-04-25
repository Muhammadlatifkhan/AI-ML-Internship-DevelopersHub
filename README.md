# 🚀 AI/ML Engineering Internship - DevelopersHub Corporation

## 👨‍💻 Intern Information
| Field | Details |
|-------|---------|
| **Name** | Muhammad Latif |
| **Role** | AI/ML Engineering Intern |
| **Submission Date** | April 25, 2026 |
| **Status** | ✅ All 6 Tasks Completed |

---

## 📊 Task Completion Overview

| Task | Topic | Key Technologies | Status |
|------|-------|------------------|--------|
| **Task 1** | Iris Dataset Visualization | Pandas, Seaborn, Matplotlib | ✅ |
| **Task 2** | Stock Price Prediction | yfinance, Scikit-learn, Random Forest | ✅ |
| **Task 3** | Heart Disease Prediction | Logistic Regression, Random Forest | ✅ |
| **Task 4** | Health Chatbot | Groq API, Llama 3.3 70B | ✅ |
| **Task 5** | Mental Health Chatbot | DistilGPT2, LoRA, PEFT | ✅ |
| **Task 6** | House Price Prediction | Gradient Boosting, Linear Regression | ✅ |

---

## 📁 Repository Structure

```
AI-ML-Internship-DevelopersHub/
│
├── Task1-Iris-Data-Visualization/
│   ├── task1_iris_visualization.py
│   ├── README.md
│   ├── requirements.txt
│   └── images/ (5 PNG files)
│
├── Task2-Stock-Price-Prediction/
│   ├── task2_stock_prediction.py
│   ├── README.md
│   ├── requirements.txt
│   └── images/ (5 PNG files)
│
├── Task3-Heart-Disease-Prediction/
│   ├── task3_heart_disease.py
│   ├── README.md
│   ├── requirements.txt
│   └── images/ (5 PNG files)
│
├── Task4-Health-Chatbot/
│   ├── task4_health_chatbot.py
│   ├── README.md
│   └── requirements.txt
│
├── Task5-Mental-Health-Chatbot/
│   ├── mental_health_chatbot_lora/ (LoRA adapters)
│   ├── task5_chatbot.py
│   ├── README.md
│   └── requirements.txt
│
├── Task6-House-Price-Prediction/
│   ├── task6_house_price.py
│   ├── README.md
│   ├── requirements.txt
│   └── images/ (5 PNG files)
│
└── README.md (this file)
```

---

## 🚀 How to Run Any Task

### 1. Clone the Repository
```bash
git clone https://github.com/Muhammadlatifkhan/AI-ML-Internship-DevelopersHub.git
cd AI-ML-Internship-DevelopersHub
```

### 2. Navigate to a Task Folder
```bash
cd Task1-Iris-Data-Visualization
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Script
```bash
python task1_iris_visualization.py
```

---

## 📈 Task Highlights & Results

### Task 1: Iris Dataset Visualization
- Exploratory Data Analysis on famous Iris dataset
- Visualizations: scatter plots, histograms, box plots, correlation heatmap
- Key insight: Petal features are more discriminative than sepal features

### Task 2: Stock Price Prediction (Apple - AAPL)
- Downloaded 2 years of historical data using yfinance
- **Linear Regression:** R² = 0.7920, MAE = $3.23
- **Random Forest:** R² = 0.4174, MAE = $5.99
- Best Model: Linear Regression

### Task 3: Heart Disease Prediction
- UCI dataset with 303 patient records
- **Logistic Regression:** 83.33% accuracy, ROC-AUC = 0.9498
- **Random Forest:** 83.33% accuracy, ROC-AUC = 0.9442
- Top features: Chest pain type (cp), Thalassemia (thal), Major vessels (ca)

### Task 4: Health Chatbot
- Groq API with Llama 3.3 70B model
- Prompt engineering for medical safety
- Safety filters for medication/dosage questions
- No local model download required

### Task 5: Mental Health Chatbot
- Fine-tuned DistilGPT2 using LoRA (parameter-efficient)
- Trainable parameters: 147,456 (only 0.04% of base model)
- Emotion detection: anxious, sad, lonely, stressed, joyful
- Runs on CPU or GPU

### Task 6: House Price Prediction
- Synthetic dataset with 1,000 houses
- **Linear Regression:** R² = 0.9453, MAE = $23,141
- **Gradient Boosting:** R² = 0.9183, MAE = $27,745
- Top feature: Square footage (68.47% importance)

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (Linear Regression, Random Forest, Gradient Boosting, Logistic Regression) |
| **Deep Learning** | PyTorch, Transformers, PEFT, LoRA |
| **APIs** | Groq API, yfinance |
| **Environment** | Python 3.11, VS Code, Google Colab |

---

## 📝 Important Notes

- **Task 4** requires a Groq API key (free at [console.groq.com](https://console.groq.com))
- **Task 5** includes fine-tuned LoRA adapters (model weights not included in GitHub due to size)
- All visualizations are saved in `images/` folders within each task
- Each task has its own `requirements.txt` for isolated dependency management

---

## 📧 Contact

For any questions about this submission, please contact:

**Muhammad Latif** - AI/ML Engineering Intern
