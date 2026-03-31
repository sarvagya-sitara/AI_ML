Fake News Detecting System


Overview
This Fake News Detection Project uses Machine Learning to classify if the news articles are REAL or FAKE. This requires implementation of Natural Language Processing (NLP).

It uses:  
 TF-IDF Vectorization to extract feature  
 Multinomial Naive Bayes Classifier  

Features
 Automatic text cleaning and preprocessing  
 Balanced dataset to train without bias  
 Predictions based on machine learning  
 Accuracy evaluated  
 User-friendly command line  
 Confidence level with each prediction  

Technology Stack
 Language: Python  
 Libraries: Pandas, Scikit-Learn Libraries  
 Concepts Used: NLP, TF-IDF, Naive Bayes  

Project Structure
├── fake_news.py  
├── news.csv  
└── README.md  

Installation and Setup
1. Clone the Repository
bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection

2. Install Dependencies
bash
pip install pandas scikit-learn


Dataset Format

 Text                              Label  
  
"Government launches scheme"        REAL    
"Aliens control Earth"              FAKE    


Running the Program  
Run the program using the following command:
bash
python fake_news.py


Program Workflow
1. Load Dataset.  
2. Preprocess & Clean Text.  
3. Convert Labels (REAL=1, FAKE=0).  
4. Balance Dataset.  
5. Train Model.  
6. Evaluate Model Accuracy.  
7. Allow User Input to check for Fake/Real News.

Example Results
Model Accuracy = 90%+, etc.

Limitations
 Quality of data used to train the model.   
 This is a basic ML model.

Future Enhancements
 Creating a web application.  
 Implementing Deep Learning Techniques.  
 Build an API.  

License  
This project is Licensed under Unlicense.  

Author  
Sohham Sarswat  
25BAC10026  
