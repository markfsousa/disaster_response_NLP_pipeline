# disaster_response_NLP_pipeline
This project is a partial requirement to graduate from the Data Science nanodegre at Udacity. The dataset was provided by Udacity in Colaboration with Figure Eight. The dataset consists of messages from social media and 36 possible labels for each message. More than one label can be assigned to each message.

A ETL pipeline was used to clean the data, transform the features into numerical values and store loaded in a database. Another pipeline consumed the data from the database to train the model to predict what should be the labels for any given message. There is also an web app that enables the input of new messages and returns the predicted labels for them.

# Development
Both pipelines, for ETL and Machine Learning, were designed with Jupyter notebook. The .ipynb file of notebooks can be found in the directories data and model, respectively.

# Dependencies
| Library               | Version |
|-----------------------|---------|
|scikit-learn           |   0.21.3|
|numpy                  |   1.17.4|
|nltk                   |   3.4.5|
|pandas                 |   0.25.3|
|sqlite                 |   3.30.1|
|sqlalchemy             |   1.3.11|
|python                 |   3.7.5|
|flask                  |   1.1.1|
|plotly                 |   4.4.1|



# Instructions:
1. Download the project:
'git clone https://github.com/markfsousa/disaster-response.git'

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
