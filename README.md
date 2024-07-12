# Email_Spam_Detection_Mchine_Learning

This project focuses on building a machine learning model to classify emails as spam or non-spam (ham). The dataset used is the SMS Spam Collection dataset, which contains SMS messages labeled as spam or ham is taken from UCI.(https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip)

## Project Overview

The project involves the following steps:

1. **Importing the Dataset**: The dataset containing SMS messages and their corresponding labels (spam or ham) is imported using pandas.

2. **Data Cleaning and Preprocessing**: Text data preprocessing techniques are applied to clean the messages and prepare them for model training. This includes removing special characters, converting text to lowercase, removing stopwords, and stemming or lemmatizing the words.

3. **Creating the Bag of Words Model**: The CountVectorizer from scikit-learn is used to convert the text data into a numerical format suitable for machine learning models.

4. **Train-Test Split**: The dataset is split into training and testing sets to evaluate the performance of the trained model.

5. **Training Model Using Naive Bayes Classifier**: The Multinomial Naive Bayes classifier is trained on the training data to classify emails as spam or ham.

6. **Model Evaluation**: The performance of the trained model is evaluated using confusion matrix and accuracy score.

## Code Overview

The project code consists of the following main components:

- **Importing the Dataset**: Pandas is used to read the dataset file (`SMSSpamCollection`) and load it into a DataFrame.

- **Data Cleaning and Preprocessing**: Regular expressions are employed to clean the text data, and NLTK is used for text preprocessing tasks such as removing stopwords and stemming/lemmatizing words.

- **Creating the Bag of Words Model**: The CountVectorizer from scikit-learn is utilized to create the bag of words model with a maximum of 5000 features.

- **Train-Test Split**: The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

- **Training Model Using Naive Bayes Classifier**: The Multinomial Naive Bayes classifier is trained on the training data using the fit method.

- **Model Evaluation**: The performance of the trained model is evaluated using confusion matrix and accuracy score metrics.

## Dependencies

- Python 3.x
- pandas
- scikit-learn
- nltk

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/email-spam-detection.git
cd email-spam-detection
