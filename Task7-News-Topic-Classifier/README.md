# Task 7: News Topic Classifier

This folder contains the files for training and deploying a BERT-based news topic classifier.

## 1. Training the Model on Google Colab

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click on **File > Upload notebook** and upload the `Train_BERT_Colab.ipynb` file from this folder.
3. Once opened, make sure you are using a GPU: Click on **Runtime > Change runtime type** and select **T4 GPU** (or any available GPU).
4. Run the cells in the notebook one by one. It will:
   - Ask for permission to mount your Google Drive (so it can save the trained model there).
   - Download the AG News dataset.
   - Fine-tune the `bert-base-uncased` model.
   - Evaluate its accuracy and F1 score.
   - Save the trained model files to a folder named `Task7_Saved_Model` in your Google Drive.

## 2. Moving the Model Files to Your PC

1. Open your [Google Drive](https://drive.google.com/).
2. Locate the folder named `Task7_Saved_Model`.
3. Download the entire folder to your local machine.
4. Extract/unzip it if necessary.
5. Inside this directory (`Task7-News-Topic-Classifier`), create a new folder named `saved_model`.
6. Copy all the contents from the downloaded `Task7_Saved_Model` folder into the `saved_model` folder you just created.
   You should end up with paths like:
   - `Task7-News-Topic-Classifier/saved_model/config.json`
   - `Task7-News-Topic-Classifier/saved_model/model.safetensors` (or `pytorch_model.bin`)
   - `Task7-News-Topic-Classifier/saved_model/vocab.txt`
   - etc.

## 3. Running the Local Streamlit Web App

1. Open your terminal or command prompt in this directory (`Task7-News-Topic-Classifier`).
2. Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. This will automatically open a tab in your web browser where you can enter news headlines and see the model's predictions live.
