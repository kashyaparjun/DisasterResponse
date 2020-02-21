import sys
import pickle

import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    """
        Function to load data from sqlite
        Input: 
            - database_filepath: That path to the sqlite DB
        Ouput:
            - X: messages
            - y: OneHotEnoded categories
            - Names of categories
    """
    
    # Read data from sql to a DataFrame
    df = pd.read_sql_table("MLStaging", "sqlite:///"+database_filepath)
    
    # Separate X and y columns
    # X - independent
    # y - dependent
    X = df["message"]
    y = df.drop(["categories", "message", "original", "genre", "id"], axis=1)
    
    print(y.columns)
    
    return X, y, list(y.columns)


def tokenize(text):
    """
        Function to tokenize the text
        Input:
            - text: The sentence that needs to be tokenized
        Output:
            - Lemmatized, lower char token array
    """
    
    # Word tokenization
    tokens = word_tokenize(text)
    
    # Initializing Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatizing, lower case conversion and trimming the words for extra space
    clean_tokens = [lemmatizer.lemmatize(i).lower().strip() for i in tokens]
    
    return clean_tokens

def build_model():
    """
        Function to build the ML model
        Input:
            -
        Ouput:
            - GridSearchCV object
    """
    
    # Forming Pipleine
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Initializing parameters for Grid search
    parameters = {
        'clf__estimator__n_estimators': [10, 50, 100]    
    }
    
    # GridSearch Object with pipeline and parameters
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Function to evaluate model
    """
    
    # Predict results of X_test
    y_pred = model.predict(X_test)

    # Converting both y_pred and Y_test into DataFrames
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    Y_test = pd.DataFrame(y_pred, columns=category_names)
    
    # Print classification report and accuracy with respect to each column
    for c in category_names:
        print(c, classification_report(Y_test[c].values, y_pred[c].values))
        print("Accuracy of "+str(c)+": "+str(accuracy_score(Y_test[c].values, y_pred[c].values)))
        


def save_model(model, model_filepath):
    """
        Function to save the ML model
    """
    
    # open the file
    pickle_out = open(model_filepath, "wb")
    
    # write model to it
    pickle.dump(model, pickle_out)
    
    # close pickle file
    pickle_out.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()