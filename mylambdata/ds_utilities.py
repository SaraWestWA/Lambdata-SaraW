import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from pdb import set_trace as breakpoint


def enlarge(n):
    """
    Parameter n is a #
    Function will muliply by 100
    """
    return n*100



def clean_frame(df):

    """
    Param is a dataframe, can have both catagorical and numeric data

    Function returns the dataframe with leading and trailing zeros removed;
    '?','', and empty cells replaced with NaN, dtype changed to float
    if possible.
    """

    df = df.applymap(lambda x: x.strip() if type(x) == str else x)
    df = df.applymap(lambda x: np.nan if type(x) == str and x == ''or
                     x == None or x == '?' else x)
    df = df.apply(pd.to_numeric, errors='ignore')
    return (df)



def null_counts(df):

    """
    Param is a dataframe, can have both catagorical and numeric data

    Function returns a dataframe of counts for  null values and 0.
    """
    # surpress warning comparing pd objects to np.nan
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    df = df.applymap(lambda x: x.strip() if type(x) == str else x)
    columns = list(df.columns)
    n_c = []
    z_c = []
    q_c = []
    m_c = []
    for i in columns:
        nan_count = df[i].isnull().sum()
        zero_count = (sum(df[i] == '0')+sum(df[i] == 0))
        q_mark_count = sum(df[i] == '?')
        missing_count = (sum(df[i] == '')+sum(df[i] == None))
        n_c.append(nan_count)
        z_c.append(zero_count)
        q_c.append(q_mark_count)
        m_c.append(missing_count)
    null_count = pd.DataFrame(data=(n_c, z_c, q_c, m_c),
                    index=['NaN', 0, '?', 'Missing'], columns=columns)
    return (null_count)

def train_validation_test_split(df, features, target,
                                train_size=0.7, val_size=0.1,
                                test_size=0.2, random_state=None,
                                shuffle=True):
    '''
    This function is a utility wrapper around the Scikit-Learn train_test_split that splits arrays or 
    matrices into train, validation, and test subsets.
    Args:
        df (Pandas DataFrame) Dataframe with code.

        X (list) : A list of features.
        y (string): A string with target column.

        FUTURE DATA
        # X (Numpy array or DataFrame): This is a dataframe (matrix) with features.
        # y (Numpy array or DataFrame): This is a pandas Series (vector) with target.

        train_size (float or int): Proportion of the dataset to include in the train split (0 to 1).
        val_size (float or int): Proportion of the dataset to include in the validation split (0 to 1).
        test_size (float or int): Proportion of the dataset to include in the test split (0 to 1).
        random_state (int): Controls the shuffling applied to the data before applying the split for reproducibility.
        shuffle (bool): Whether or not to shuffle the data before splitting

    Returns:
        Train, test, and validation dataframes for features (X) and target (y). 
    '''
    '''
    # X_train_val, X_test, y_train_val, y_test = train_test_split(
    #     self.X, self.y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    '''
    X = df[features]
    y = df[target]


    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (train_size + val_size),
        random_state=random_state, shuffle=shuffle)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    # Simple test for enlarger function
    # print(enlarge(8))

    # Test for train_validation_test_split function
    raw_data = load_wine()
    df = pd.DataFrame(data=raw_data['data'], columns=raw_data['feature_names'])
    df['target'] = raw_data['target']

    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(df, features=['ash', 'hue'], target='target')
    
    # breakpoint()

    

