# 📧 **Email Spam Filtering**
## 📝 **Overview**
The goal of this project is to develop an Email Spam Classifier using Machine Learning techniques. The model is trained to distinguish between spam and ham (non-spam) emails by analyzing textual content.

## 📚 **Libraries Used**
To build and evaluate the spam classifier, the following libraries are utilized:

- **Pandas:** For data manipulation and preprocessing.
- **NumPy:** Helps with numerical operations and data handling.
- **Matplotlib & Seaborn:** Used for visualizing data distributions.
- **Scikit-learn:** Machine learning library for model building, training, and evaluation.

## 🌍 **Dataset Information**
The dataset used for this project contains thousands of emails labeled as either spam or ham. The dataset includes:

- **Email text:** The raw content of emails.
- **Label:** Indicates whether an email is spam (1) or ham (0).
- The dataset is sourced from Kaggle and is cleaned before use.

## 🔧 **Feature Engineering**
Feature engineering involves converting email text into a format suitable for machine learning. This includes:

- **Tokenization:** Splitting emails into individual words.
- **Lowercasing & Punctuation Removal:** Standardizing text format.
- **Stopword Removal:** Removing common words like "the," "and," etc.
- **TF-IDF Vectorization:** Converting text into numerical representations based on word importance.

## 🧠 **Machine Learning Models**
Several machine learning algorithms are explored for email classification:

- **Logistic Regression:** A simple yet effective classifier for binary classification.
- **Naïve Bayes Classifier:** Performs well on text data by leveraging probabilistic models.
- **Random Forest Classifier:** An ensemble method for improving accuracy.

## 🔄 **Data Preprocessing**
To ensure high model performance, the following steps are applied:

- **Handling Missing Data:** Removing or filling missing email texts.
- **Text Normalization:** Converting words to their root form.
- **Splitting Data:** Dividing into training (80%) and testing (20%) sets.

## 📊 **Model Evaluation**
The trained models are evaluated using key metrics:

- **Accuracy Score:** Measures the percentage of correct predictions.
- **Precision & Recall:** Evaluates the trade-off between false positives and false negatives.
- **F1-Score:** A balanced measure of model performance.

## 📉 **Visualization**
Data visualization helps in understanding patterns and results:

- **Spam vs Ham Distribution:** A bar chart showing label frequencies.
- **Word Cloud:** Displays frequently used words in spam and ham emails.
- **Confusion Matrix:** Helps assess model prediction errors.

## 🔚 **Conclusion**
- This project successfully classifies emails as spam or ham using machine learning models. The implementation of TF-IDF and Naïve Bayes provides high accuracy (96.2%). Future improvements can include:
- Deep Learning models for enhanced text classification.
- Real-time spam filtering for live email processing.
- Integration with email platforms for practical applications.
