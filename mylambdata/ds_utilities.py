import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from pdb import set_trace as breakpoint
from IPython.display import display


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
                            index=['NaN', 0, '?', 'Missing'],
                            columns=columns)
    return (null_count)

class My_Data_Splitter():
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.X = df[features]
        self.y = df[target]


    def train_validation_test_split(self,
                                train_size=0.7, val_size=0.1,
                                test_size=0.2, random_state=None,
                                shuffle=True):


        X_train_val, X_test, y_train_val, y_test = train_test_split(
                    self.X, self.y, test_size=test_size, random_state=random_state,
                    shuffle=shuffle)

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                    test_size=val_size / (train_size + val_size),
                                    random_state=random_state, shuffle=shuffle)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def print_split_summary(self, X_train, X_val, X_test):

        print('########################## TRAINING DATA ##########################')
        print(f'X_train Shape: {X_train.shape}')
        display(X_train.describe(include='all').transpose())
        print('')

        print('########################## VALIDATION DATA ##########################')
        print(f'X_val Shape: {X_val.shape}')
        display(X_val.describe(include='all').transpose())
        print('')

        print('########################## TEST DATA ##########################')
        print(f'X_test Shape: {X_test.shape}')
        display(X_test.describe(include='all').transpose())
        print('')



if __name__ == '__main__':
    # Simple test for enlarger function
    # print(enlarge(8))

    # Test for train_validation_test_split function
    raw_data = load_wine()
    df = pd.DataFrame(data=raw_data['data'], columns=raw_data['feature_names'])
    df['target'] = raw_data['target']

    #  breakpoint()

    #X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split
        #(df, features=['ash', 'hue'], target='target')

    # Test My_Data_Splitter
    splitter = My_Data_Splitter(df=df, features=['ash', 'hue'], target='target')
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_validation_test_split()
    splitter.print_split_summary(X_train, X_val, X_test)
