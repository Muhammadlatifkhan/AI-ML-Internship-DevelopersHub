# Task 1: Iris Dataset - Exploratory Data Analysis (EDA)

## 🎯 Objective
Load, inspect, and visualize the Iris dataset to understand data trends, feature relationships, and distributions using Python in VS Code.

## 📊 Dataset
**Iris Dataset** (built-in from seaborn library)
- **Samples:** 150 flowers
- **Features:** 4 numerical (sepal_length, sepal_width, petal_length, petal_width)
- **Target Classes:** 3 species (setosa, versicolor, virginica)
- **Missing Values:** None

## 🛠️ Tools & Technologies
- **VS Code** - Development environment
- **Python 3.9+** - Programming language
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **NumPy** - Numerical operations

## 📁 Project Structure
Task1-Iris-Data-Visualization/
├── task1_iris_visualization.py # Main Python script
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── data/ # Dataset folder (optional)
└── images/ # Saved visualizations
├── scatter_plots.png
├── histograms.png
├── box_plots.png
├── correlation_heatmap.png
└── pairplot.png

## 🚀 How to Run

1. Install Dependencies
Open terminal in VS Code and run:
```bash
pip install -r requirements.txt```
2. Execute the Script
```bash
python task1_iris_visualization.py```
3. View Output
Terminal: Prints data statistics and insights

Pop-up Windows: Shows interactive visualizations

Images Folder: Contains saved plots (PNG format)

📈 Visualizations Included
Visualization	Purpose	File Name
Scatter Plots	Feature relationships	scatter_plots.png
Histograms	Value distributions	histograms.png
Box Plots	Outlier detection	box_plots.png
Correlation Heatmap	Feature correlations	correlation_heatmap.png
Pairplot	Complete overview	pairplot.png
🔍 Key Findings
Setosa species is clearly separable from Versicolor and Virginica

Petal length and width have strong positive correlation (0.96)

Virginica has the largest petal dimensions (avg petal length: 5.55cm)

No missing values or significant outliers detected

Petal features are more discriminative than sepal features

📊 Sample Output (Terminal)
text
============================================================
TASK 1: IRIS DATASET EDA
============================================================
✅ Dataset loaded successfully!
📊 Shape: (150, 5)
📋 Columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

First 5 rows:
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
...
✅ Task Completion Checklist
Load dataset using pandas

Print shape, columns, and head()

Use info() and describe() for statistics

Create scatter plots (feature relationships)

Create histograms (value distributions)

Create box plots (outlier detection)

Save all visualizations

Document key findings

📝 Notes
All visualizations are automatically saved to the images/ folder

Script runs completely offline (no API keys needed)

Total execution time: ~5-10 seconds

👨‍💻 Author
AI/ML Engineering Intern - DevelopersHub Corporation

📅 Date
April 17, 2026




