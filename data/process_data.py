# Basic python import
import sys

# Data Manipulation imports
import pandas as pd
import numpy as np

# DB import
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        Function to Load Messages and Categories
        Input:
            - messages_filepath: path to messages file
            - categories_filepath: path to categories file
        Output:
            - cancated DataFrame of messages and categories
    """
    
    # Loading messages
    messages = pd.read_csv(messages_filepath)
    
    # Loading categories
    categories = pd.read_csv(categories_filepath)
    
    # Concating both DataFrames
    return pd.concat([messages, categories], axis=1)


def clean_data(df):
    """
        Function to clean Data
        Input: 
            - df: DataFrame that contains both messages and categories
        Ouput:
            - df: DataFrame that has OneHotEncoded Categories
    """
    
    # Separating categories
    categories = df['categories'].str.split(";", expand=True)
    
    # Getting list of columns in categories
    categories.columns = [i.split("-")[0] for i in categories.iloc[0,:].values]
    
    # For each category, separating the int value
    for column in categories:
        categories[column] = categories[column].str.split("-").str[1]
        categories[column] = categories[column].astype("int64")
        
    # Dropping the original categories column on df
    df.drop(["categories"], axis=1)
    
    # Adding the OntHotEncoded columns into df
    df = pd.concat([df, categories], axis=1)
    
    # Dropping duplicate values
    df = df.drop_duplicates()
    
    return df
    


def save_data(df, database_filename):
    """
        Saving DataFrame to SQLite DB
        Input:
            - df: DataFrame containing messages and categories
            - database_filename: filename for the SQLite Database
    """
    
    # SQLite engine
    engine = create_engine("sqlite:///"+database_filename)
    
    # Writing to SQLite DB
    df.to_sql('MLStaging', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()