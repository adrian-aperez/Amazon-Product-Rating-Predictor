# Product Rating Predictor


![Screenshot](/assets/Screenshot2.png) 
*Screenshot of the Streamlit interface*

![Demo GIF](https://i.imgur.com/5XbJQ2F.gif)  

[🔗 Live Demo](https://amazon-rating-predictor.streamlit.app/)

## Description

**Product Rating Predictor** is a machine learning application designed to predict Amazon product ratings based on product features (trained on men's hygiene products). 

It uses two specialized models:
- **Modelo_bajo**: For products with **<60 reviews**
- **Modelo_alto**: For products with **>60 reviews**

## Project Overview

1. **Data Pipeline**:
   - Web scraping & external data collection
   - Data cleaning & preprocessing

2. **Exploratory Analysis (EDA)**:
   - Statistical insights
   - Business intelligence visualizations

3. **Feature Engineering**:
   - Custom data transformers
   - NLP processing (TF-IDF for text features)

4. **Model Development**:
   - Machine Learning (Scikit-learn)
   - Deep Learning experiments
   - Hyperparameter tuning & cross-validation

5. **Deployment**:
   - Streamlit web application
   - Model serving & inference

## Key Features

| Feature          | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| 🌐 Responsive    | Mobile-friendly interface                                                   |
| ⚡ Fast Predictions | Optimized for quick inference (<1s)                                      |
| ⭐ Dual-Model System | Specialized models for high/low-review products                         |
| 📊 Business Insights | Includes analytical visualizations                                      |
| 🔍 Explainable AI | Feature importance analysis available                                     |
| 🚀 Production-Ready | Deployed with Streamlit                                                  |

## Tech Stack
🛠️ Core: Python, Pandas, NumPy

🤖 ML: Scikit-learn, PyCaret, TensorFlow (Keras)

📊 Visualization: Matplotlib, Seaborn

📝 NLP: NLTK, TF-IDF

🚀 Deployment: Streamlit, Joblib



## Installation
  
### Clone repository
```
git clone https://github.com/adrian-aperez/Product-Rating-Predictor.git
```
### Navigate to project
```
cd Product-Rating-Predictor
```
### Install dependencies
```
pip install -r requirements.txt
```
### Launch Streamlit app
```
python -m streamlit run App.py
```

### Notes:
1. **Screenshot/GIF**:  
   - I used placeholder Imgur links (replace with your actual screenshots/GIFs).  
   - To capture these:  
     - Screenshot: Use Snip & Sketch (Windows) or Screenshot (Mac).
     - GIF: Record with ScreenToGif (Windows) or LICEcap (Mac).
     

2. **Contributing Section**:  
   - Standard GitHub workflow  
   - Encourages collaboration while maintaining code quality  

Want me to adjust any section? For example:  
- Add a "Roadmap" for future features  
- Include contributor badges  
- Expand testing guidelines
