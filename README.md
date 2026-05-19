# 🚀 AI/ML Engineering Internship - DevelopersHub Corporation

## 👨‍💻 Intern Information

| Field | Details |
|-------|---------|
| Name | Muhammad Latif |
| Role | AI/ML Engineer |
| Submission Date | May 19, 2026 |
| Status | ✅ All Tasks Completed |

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
| **Task 7** | News Topic Classifier | BERT, Transformers, Streamlit | ✅ |
| **Task 8** | Customer Churn Pipeline | scikit-learn Pipeline, GridSearchCV | ✅ |
| **Task 9** | Multimodal Price Predictor | ResNet18, PyTorch, Feature Fusion | ✅ |
| **Task 10** | RAG Chatbot | LangChain, Groq, ChromaDB | ✅ |
| **Task 11** | Auto Ticket Tagging | LLM, Zero/Few-shot Learning | ✅ |

---

## 📁 Repository Structure

```
AI-ML-Internship-DevelopersHub/
│
├── Task1-Iris-Data-Visualization/
│   ├── task1_iris_visualization.py
│   ├── README.md
│   └── requirements.txt
│
├── Task2-Stock-Price-Prediction/
│   ├── task2_stock_prediction.py
│   ├── README.md
│   └── requirements.txt
│
├── Task3-Heart-Disease-Prediction/
│   ├── task3_heart_disease.py
│   ├── README.md
│   └── requirements.txt
│
├── Task4-Health-Chatbot/
│   ├── task4_health_chatbot.py
│   ├── README.md
│   └── requirements.txt
│
├── Task5-Mental-Health-Chatbot/
│   ├── task5_chatbot.py
│   ├── README.md
│   └── requirements.txt
│
├── Task6-House-Price-Prediction/
│   ├── task6_house_price.py
│   ├── README.md
│   └── requirements.txt
│
├── Task7-News-Topic-Classifier/
│   ├── app.py
│   ├── train_multimodal.py
│   ├── saved_model/
│   ├── README.md
│   └── requirements.txt
│
├── Task8-Customer-Churn-Pipeline/
│   ├── train_pipeline.py
│   ├── predict.py
│   ├── outputs/
│   ├── README.md
│   └── requirements.txt
│
├── Task9-Housing-Price-Prediction/
│   ├── train_multimodal.py
│   ├── data/
│   ├── outputs/
│   ├── README.md
│   └── requirements.txt
│
├── Task10-RAG-Chatbot/
│   ├── chatbot.py
│   ├── ingest_documents.py
│   ├── vector_store/
│   ├── README.md
│   └── requirements.txt
│
├── Task11-Support-Ticket-Tagger/
│   ├── ticket_tagger.py
│   ├── evaluate_models.py
│   ├── data/
│   ├── README.md
│   └── requirements.txt
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
cd Task7-News-Topic-Classifier
```

### 3. Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
# For Streamlit apps (Task 7, 10, 11)
streamlit run app.py

# For Python scripts (Task 1-6, 8, 9)
python train_pipeline.py
```

---

## 📈 Task Highlights & Results

### Task 1: Iris Dataset Visualization
- Exploratory Data Analysis on Iris dataset
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
- Top features: Chest pain type, Thalassemia, Major vessels

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

### Task 7: News Topic Classifier (BERT)
- Fine-tuned BERT-base-uncased on AG News dataset
- **Accuracy:** 94.64%, **F1 Score:** 94.65%
- Categories: World, Sports, Business, Sci/Tech
- Streamlit web interface for real-time predictions

### Task 8: Customer Churn Pipeline
- End-to-end ML pipeline with scikit-learn
- **Best Model:** Random Forest (tuned)
- **Accuracy:** 80.13%, **ROC-AUC:** 84.27%
- GridSearchCV for hyperparameter optimization
- Production-ready pipeline exported with joblib

### Task 9: Multimodal Price Predictor
- Combines CNN (ResNet18) + Tabular data
- Feature fusion architecture
- **MAE:** $267,023, **RMSE:** $322,870
- Demonstrates multimodal learning concepts

### Task 10: RAG Chatbot
- LangChain + Groq + ChromaDB
- Conversation memory for context
- Document retrieval from knowledge base
- **LLM:** Llama 3.3 70B (10x faster)

### Task 11: Auto Ticket Tagging
- Zero-shot vs Few-shot learning comparison
- **Categories:** network_issues, billing, account_access, technical_bug, subscription, performance, api_support
- Top-3 predictions with confidence scores
- Streamlit web interface

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Classical ML** | Scikit-learn (Linear Regression, Random Forest, Gradient Boosting, Logistic Regression) |
| **Deep Learning** | PyTorch, Transformers, PEFT, LoRA, BERT |
| **Computer Vision** | ResNet18, CNNs, Transfer Learning |
| **LLM & RAG** | Groq API, LangChain, ChromaDB, Llama 3.3 70B |
| **Web Deployment** | Streamlit |
| **APIs** | Groq API, yfinance |
| **Environment** | Python 3.11, VS Code, Google Colab, Jupyter |

---

## 📊 Skills Demonstrated

| Skill Area | Tasks |
|------------|-------|
| **Data Analysis & Visualization** | Task 1, 2, 3, 6 |
| **Classical Machine Learning** | Task 2, 3, 6, 8 |
| **Deep Learning (NLP)** | Task 5, 7, 10, 11 |
| **Computer Vision** | Task 9 |
| **Multimodal AI** | Task 9 |
| **LLM & RAG Applications** | Task 4, 10, 11 |
| **ML Pipelines** | Task 8 |
| **Model Deployment** | Task 7, 8, 9, 10, 11 |
| **Prompt Engineering** | Task 4, 10, 11 |
| **Version Control** | Git, GitHub |

---

## 🔑 API Keys Required

| Task | API Key | Where to Get |
|------|---------|--------------|
| Task 4 | Groq API | [console.groq.com](https://console.groq.com) |
| Task 10 | Groq API | [console.groq.com](https://console.groq.com) |
| Task 11 | Groq API | [console.groq.com](https://console.groq.com) |

**Note:** All Groq API keys are free (no credit card required).

---

## 📝 Important Notes

- **Tasks 4, 10, 11** require Groq API key in `.env` file
- **Task 7** requires downloading trained model (instructions in task README)
- **Task 9** uses synthetic images (no external download needed)
- **Task 5** includes LoRA adapters (fine-tuned weights)
- Each task has its own `requirements.txt` for isolated dependency management
- Virtual environments recommended for each task

---

## 🎯 Learning Outcomes

Throughout this internship, I have gained proficiency in:

1. **Data Science:** Exploratory data analysis, visualization, feature engineering
2. **Machine Learning:** Regression, classification, ensemble methods, pipelines
3. **Deep Learning:** Transformers, BERT, CNNs, transfer learning, LoRA fine-tuning
4. **Computer Vision:** Image feature extraction, multimodal learning
5. **LLM Applications:** Prompt engineering, RAG, few-shot learning
6. **MLOps:** Model deployment, Streamlit, version control
7. **Production Readiness:** ML pipelines, model serialization, API integration

---

## 📧 Contact

Muhammad Latif  
AI/ML Engineer

- GitHub: [github.com/Muhammadlatifkhan](https://github.com/Muhammadlatifkhan)
- Email: laahmad7777@gmail.com
- LinkedIn:[linkedin.com/in/muhammad-latif-khan](https://linkedin.com/in/your-profile)

---

## 🙏 Acknowledgments

- **DevelopersHub Corporation** for this internship opportunity
- **MSA Technologies** for guidance and mentorship
- **Groq** for providing free LLM API access
- **Hugging Face** for transformers and datasets
- **OpenAI, Meta** for open-source models (BERT, Llama)

---

## 📅 Project Status

✅ **COMPLETED** - May 19, 2026



---

⭐ **If you found this repository helpful, please give it a star on GitHub!**
```
