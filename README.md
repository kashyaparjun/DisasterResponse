# Disaster Response Pipeline Project

## Description

This is a project to create a disaster reponse pipeline using the Firgure Eight dataset. The dataset contains pre-labeled messages from an actual disaster. The labels are assigned with an intent to not only categorize the message as a request or not but also, it is used to classify and check if it is a call for help so that appropriate action can be taken.

This project is divided into 3 parts:
1. Data Cleaning and preprocessing, an ETL pripeline is created clean the data and store them in a structured database (SQLite)
2. A Machine learning pipeling to classify the messages amongst the present 36 categories
3. A Web app to display the results

## Installation and Usage

### Dependencies
1. Interpreter: Python 3.6+
2. Processing libraries: Numpy, Pandas, Scikit-learn, NLTK
3. DB connect library: SQLalchemy
4. For Web app: Flask, Ploty

### Usage
1. Clone the repo using:
```
git clone https://github.com/kashyaparjun/DisasterResponse.git
```
2. Download the saved model pickle file from the below link, unzip it and put it inside the models folder:
```
https://drive.google.com/file/d/1Oo2fwvLpkIVrS-6f2twfmLxSvQ5ilvrY/view?usp=sharing
```
2. Run the following commands in the DisasterResponse directory:
    - ETL pipeline to import data, clean it and store into SQLite DB:
    ```
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DataResponse.db
    ```
    - ML pipeline and saving the model (This might take a while to run, be patient!):
    ```
    python model/train_classifier.py data/DataResponse.db models/classifier.pkl
    ```
3. Finally, run the web app by running:
```
python app/run.py
```
4. Go the URL:
```
http://0.0.0.0:3001
```

## License
MIT

## Acknowledgements
1. Figure Eight for providing pre-labeled messages dataset

