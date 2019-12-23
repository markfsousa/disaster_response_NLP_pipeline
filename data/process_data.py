import sys
import pandas as pd
from sqlalchemy import create_engine

# create a dataframe of the 36 individual category columns
def expand_categories(df_cat, column_name, col_sep=';', value_sep='-'):
    """ Expand in columns the categories stored in a single column.
    
    Parameters
    ----------
    df_cat: DataFrame
        The categories data.
    column_name: str
        The name of the column storing the categories.
    col_sep: str
        The separator used to separate the categories from the single column.
    value_sep:
        The separator used to split the category name from the value of category;
    """

    # create a dataframe of the 36 individual category columns
    categories = df_cat[column_name].str.split(col_sep, expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    
    # use this row to extract a list of new column names for categories.
    categories.columns = row.str.split(value_sep, expand=True)[0]
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split(value_sep, expand=True)[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    return categories

def load_data(messages_filepath, categories_filepath, index_name, cat_column):
    """ Load Data to be processed
    
    Parameters:
    -----------
        messages_filepath: str
            Path to messages csv file
        categories_filepath: str
            Path to categories csv file
    Output
    ------
        The DataFrame loaded with data from messages and categories
    """

    # load messages dataset
    df_msgs = pd.read_csv(messages_filepath, index_col=index_name)
    print('OK, messages loaded!')
    
    # load categories dataset
    df_cat =  pd.read_csv(categories_filepath, index_col=index_name)
    print('OK, categories loaded!')
    
    # Expand categories
    df_cat_exp = expand_categories(df_cat, column_name=cat_column)

    # Merge categories and messages
    df_merge = df_msgs.merge(df_cat_exp, left_index=True, right_index=True, how='inner')
    df_merge = df_merge.reset_index() # After merge on id, use a more reliable index

    return df_merge


def clean_data(df, drop_columns, predictor_vars, replacements):
    """ Clean the DataFrame.
    
    Remover duplicated values, undesired columns, NaNs, and replace values.
    
    Parameters
    ----------
    df: DataFrame
        The dataframe to be cleaned.
    drop_columns: list of str
        Columns to be dropped.
    predictor_vars: list of strs
        Variables to check if there are repetitions.
    replacements: dict {str : (old, new)}
        The dictionary with the columns to apply the replacement and a tuple
        with the old value to be replaced and the new value.
    """
    
    print("Dropping columns:", drop_columns)
    df_clean = df.drop(columns=drop_columns)
    
    # Drop identical rows
    df_clean = df_clean.drop_duplicates()
    
    # Drop rows with same predictor values with different predicted values
    df_clean = df_clean.drop_duplicates(subset=predictor_vars, keep=False)
    
    df_clean = df_clean.dropna()
    
    # Replace values in columns
    for col, rep in replacements:
        df_clean[col] = df_clean[col].replace(rep[0], rep[1])
    return df_clean


def save_data(df, database_filename, table_name):
    """ Save the dataframe into the database.
    
    Parameters
    ----------
    df: DataFrame
        The data to be stored.
    database_filename: str
        The name of the database.
    table_name: str
        The name of the table to store the data.
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(table_name, engine, if_exists='replace')
    pass  

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        index_name = 'id' # Index column in csv file
        drop_columns = ['original', 'child_alone'] # Columns to be droped
        # Although 'genre' is used in predictions, it is used in a visualization in the web app.
        
        cat_column = 'categories' # Categories column
        replacements=[('related', (2,1))] # Replace 2 by 1 in the column 'related'
        predictor_vars = ['message'] # Independent variables
        table_name = 'Messages_Categories'# Table to store the results

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath, index_name, cat_column)

        print('Cleaning data...')
        df = clean_data(df, drop_columns, predictor_vars, replacements=replacements)
        

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, table_name=table_name)

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