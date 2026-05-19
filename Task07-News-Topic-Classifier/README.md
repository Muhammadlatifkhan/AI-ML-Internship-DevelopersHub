# Task 7: News Topic Classifier
## Prerequisites

- Python 3.8 or higher
- Internet connection for training (Colab)
- 8+ GB RAM for local inference
This folder contains the files for training and deploying a BERT-based news topic classifier.

## 1. Training the Model on Google Colab

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click on **File > Upload notebook** and upload the `Train_BERT_Colab.ipynb` file from this folder.
3. Once opened, make sure you are using a GPU: Click on **Runtime > Change runtime type** and select **T4 GPU** (or any available GPU).
4. **Run all cells in order.** The notebook will:
   - Install required libraries (transformers, datasets, etc.)
   - Download the AG News dataset
   - Fine-tune the `bert-base-uncased` model
   - Evaluate accuracy and F1 score
   - **Save the model locally in Colab and automatically download it as a ZIP file to your computer**

**Note:** The notebook does NOT require Google Drive mounting. The trained model downloads directly to your PC as `news_classifier_model.zip`.

## 2. Extracting and Setting Up the Model Files

1. After training completes, the file `news_classifier_model.zip` will automatically download to your computer.
2. Extract the ZIP file (right-click > Extract All).
3. Inside the extracted folder, navigate to `content/saved_model/`.
4. Copy ALL files from `content/saved_model/` into this directory's `saved_model` folder:
   - Create `saved_model` folder if it doesn't exist
   - You should have files like:
     - `saved_model/config.json`
     - `saved_model/model.safetensors` (or `pytorch_model.bin`)
     - `saved_model/tokenizer.json`
     - `saved_model/tokenizer_config.json`
     - `saved_model/vocab.txt`

## 3. Running the Local Streamlit Web App

1. Open terminal/command prompt in this directory (`Task7-News-Topic-Classifier`):
   ```bash
   cd Task7-News-Topic-Classifier
   ```

2. (Optional) Create and activate virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Your browser will automatically open to `http://localhost:8501` where you can:
   - Enter news headlines
   - Click "Classify"
   - See predictions (World, Sports, Business, Sci/Tech) with confidence scores

## Training Results

The fine-tuned BERT model achieved:
- **Accuracy:** 94.64%
- **F1 Score:** 94.65%

## Model Performance Examples

| Headline | Prediction | Confidence |
|----------|------------|------------|
| "NASA's Perseverance rover discovers organic molecules on Mars" | Sci/Tech | 98.30% |
| "Earthquake hits Japan, tsunami warnings issued" | World | 99.89% |
| "Tesla stock surges after record deliveries" | Business | 99.46% |
| "Argentina wins World Cup final" | Sports | 96.69% |

## Troubleshooting

### If Streamlit shows "Unable to deploy":
- Ignore the cloud deployment panel
- Use incognito/private browser window
- Go directly to `http://localhost:8501`

### If model not found error:
- Ensure `saved_model/` folder exists in this directory
- Verify it contains `config.json` and model weight files

### If torchvision errors appear:
These warnings are safe to ignore - they don't affect the text classifier.

## Files in This Folder

- `app.py` - Streamlit web application
- `requirements.txt` - Python dependencies
- `Train_BERT_Colab.ipynb` - Colab training notebook (no Drive mounting required)
- `saved_model/` - Folder containing trained model files (create after training)

## Notes

- The model file (`model.safetensors`) is ~437MB and not included in GitHub
- Train using the provided Colab notebook to generate your own model
- The app runs completely offline once model files are in place
```
## Author & Contact

**Muhammad Latif**

- 📧 Email: laahmad7777@gmail.com
- 🔗 GitHub: https://github.com/Muhammadlatifkhan
- 💼 LinkedIn: www.linkedin.com/in/mxlatif
- 🌐 Portfolio: https://github.com/Muhammadlatifkhan/AI-ML-Internship-DevelopersHub/tree/main/Task7-News-Topic-Classifier

## License

This project is for educational purposes as part of an AI/ML internship.

## Acknowledgements

- **AG News Dataset** - For providing the news classification dataset
- **Hugging Face Transformers** - For the BERT model implementation
- **Google Colab** - For providing free GPU resources for training

---

📅 Last Updated: May 2026





