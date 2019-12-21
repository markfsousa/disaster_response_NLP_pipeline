import sys
import datetime
import numpy as np
import gzip
import pandas as pd
import pickle
import re
import time

from sqlalchemy import create_engine
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from workspace_utils import active_session

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath, table_name):
    """Load messages and categories from the database.
    
    Arguments
    ---------
        database_filepath: str
            path to SQLite db
        table_name: str
            Table from database to read the messages
    Output
    ------
        X: DataFrame
            Preditor (messages)
        Y: DataFrame
            Predictable variable (categories)
        category_names: list
            Available categories labels.
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name, engine, index_col='index')
    X = df['message']
    y = df.drop(columns=['id', 'message'])
    return X, y, y.columns.values

#def build_tokenizer():
stopwords_english = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
lemmatize = lemmatizer.lemmatize

def tokenize(text):
    """ Tokenize a message.
    
    Replace url by the constant value 'url_placeholder', remove pontuation, 
    ormalize case, remove english stopwords, and lematize.
    
    Arguments
    ---------
        text: str
            Text to be tokenized
    Output
    ------
        tokens: list
            Tokenized text.
    """

    # Replace urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, "url_placeholder", text)

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = nltk.word_tokenize(text)

    # lemmatize andremove stop words (return words not in stopwords)
    tokens = [lemmatize(word) for word in tokens if word not in stopwords_english]

    return tokens
    
#    return tokenize

class MeanStack:
    """ Class to apply fit and predict function to an ensamble of methods.
    """
    
    def __init__(self, models):
        self.models = models
    
    def fit(self, X_train, y_train):
        """ Fit all the models.
        
        Parameters
        ----------
        X_Train: DataFrame
            The data to train the classifiers.
        y_train:
            The desired target labels.
        """
        
        for model in self.models:
            t0 = datetime.datetime.now()
            print('Trainig:', type(model).__name__, t0)
            print('X_train.shape', X_train.shape)
            print('y_train.shape', y_train.shape)
            model.fit(X_train, y_train)
            
            print(type(model).__name__, ' training time:', datetime.datetime.now() - t0)
            
            
    
    def predict(self, X):
        """ Make the predictions for the given X.
        
        Parameters
        ----------
        X
            The predictor variables.
        """
        
        preds = [model.predict(X) for model in self.models]
        preds_mean = np.mean(preds, axis=0)
        preds_result = np.array(preds_mean >= 0.5, dtype='int32')
        
        return preds_result
        
    
def build_model():
    """
    Build the classifier model.
    
    Output
    ------
    A Scikitlearn Pipeline to process the messages, fit, and make predictions.
    """
    
    clf = MultiOutputClassifier(AdaBoostClassifier())

    pipeline = Pipeline([
        ('counts', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', clf)
    ])
    
    return pipeline


def evaluate_model_classes(y_test, preds, category_names):
    """ Evaluates the classifier according to each category.
    
    Parameters
    ----------
    y_test:
        Expected values.
    preds:
        Predicted values.
    category_names: list
        Labels for the set of categories.
    """
    
    # Print header
    print('='*70)
    print('{:>25} {:>10} {:>10} {:>10} {:>10}'.format('Category', 'F1', 'Precision', 'Recall', 'Accuracy'))
    print('-'*70)
    
    # Print score values
    for i, cat_name in enumerate(category_names):
        
        cat_test = y_test.iloc[:, i]
        cat_pred = preds[:, i]
        
        f1 = f1_score(cat_test, cat_pred)
        prec = precision_score(cat_test, cat_pred)
        rec = recall_score(cat_test, cat_pred)
        acc = accuracy_score(cat_test, cat_pred)
        
        print('{:>25} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}'.format(cat_name, f1, prec, rec, acc))
        
    print('='*70)

def evaluate_overall_results(y_test, y_pred):
    """ Evaluates the classifier according to overall categories.
    
    Parameters
    ----------
    y_test:
        Expected values.
    preds:
        Predicted values.
    """
    
    # Compute scores
    acc        = accuracy_score(y_test, y_pred)
    f1_mic     = f1_score(y_test, y_pred, average='micro')
    f1_mac     = f1_score(y_test, y_pred, average='macro')
    f1_wei     = f1_score(y_test, y_pred, average='weighted')
    f1_sam     = f1_score(y_test, y_pred, average='samples')
    prec_mic   = precision_score(y_test, y_pred, average='micro')
    prec_mac   = precision_score(y_test, y_pred, average='macro')
    prec_wei   = precision_score(y_test, y_pred, average='weighted')
    prec_sam   = precision_score(y_test, y_pred, average='samples')
    recall_mic = recall_score(y_test, y_pred, average='micro')
    recall_mac = recall_score(y_test, y_pred, average='macro')
    recall_wei = recall_score(y_test, y_pred, average='weighted')
    recall_sam = recall_score(y_test, y_pred, average='samples')
    
    # Print Metrics names
    template = '|{:>8}  ' + ('|{:^31}' * 3) + '|'
    print(template.format('Accuracy', 'F1', 'Precision', 'Recall'))
    
    # Print average types
    template = '|{0:<8}{1:<8}{2:<8}{3:<7}' * 3
    print('          ', template.format('Micro', 'Macro', 'Weight', 'Samples'))
    
    #Print metrics values
    template = '{:<8.3}'*13
    print('   ', template.format(acc,
                          f1_mic,
                          f1_mac,
                          f1_wei,
                          f1_sam,
                          prec_mic,
                          prec_mac,
                          prec_wei,
                          prec_sam,
                          recall_mic,
                          recall_mac,
                          recall_wei,
                          recall_sam))
    return acc

def save_model(model, model_filepath):
    """ Save the model
    
    Parameters
    ----------
    model:
        The classifier or pipeline to save.
    model_filepath:
        Path to save the file.
    """
    file = gzip.GzipFile(model_filepath, 'wb')
    file.write(pickle.dumps(model, 1))
    file.close()
    #pickle.dump(model, open(model_filepath, "wb") )



def main():
    if len(sys.argv) == 3:
        with active_session():
        
            database_filepath, model_filepath = sys.argv[1:]
            table_name = 'Messages_Categories'
    
            print('Loading data...\n    DATABASE: {}'.format(database_filepath))

            X, y, category_names = load_data(database_filepath, table_name)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
            print('Building model...')
            model = build_model()

            print('Training model...')
            model.fit(X_train, y_train)
            #model = pickle.load( open('model_try_2019-12-20 16_48_59.352060.pkl', "rb" ) )

            print('Evaluating model...')
            y_preds = model.predict(X_test)
            evaluate_overall_results(y_test, y_preds)
            evaluate_model_classes(y_test, y_preds, category_names)

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