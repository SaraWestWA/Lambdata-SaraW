import pandas as pd
import numpy as np
import warnings

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

        '''
        This function is a utility wrapper around the Scikit-Learn train_test_split that splits arrays or 
        matrices into train, validation, and test subsets.
        Args:
            X (Numpy array or DataFrame): This is a dataframe with features.
            y (Numpy array or DataFrame): This is a pandas Series with target.
            train_size (float or int): Proportion of the dataset to include in the train split (0 to 1).
            val_size (float or int): Proportion of the dataset to include in the validation split (0 to 1).
            test_size (float or int): Proportion of the dataset to include in the test split (0 to 1).
            random_state (int): Controls the shuffling applied to the data before applying the split for reproducibility.
            shuffle (bool): Whether or not to shuffle the data before splitting
        Returns:
            Train, test, and validation dataframes for features (X) and target (y).
        '''

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

class My_Ready_Frame():
    def __init__(self, df):
        self.df = df
    """
    Param df = dataframe, can have both catagorical and numeric data
    """

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

    def null_counts(df, npnan, zero, qmark, missing):

        """
        Param (df = dataframe, columns = df(columns), npnan=True(default),
        zero=True(default), qmark=True(default), missing=True(default)).

        Function returns a dataframe of counts for specified null values and 0,
        default is all.
        """
        # surpress warning comparing pd objects to np.nan
        warnings.simplefilter(action='ignore', category=FutureWarning)
        columns= list(df.columns)
        dfn = df.applymap(lambda x: x.strip() if type(x) == str else x)
        keeper = []
        index = []
        if npnan == True:
            n_c = []
            for i in columns:
                nan_count = dfn[i].isnull().sum()
                n_c.append(nan_count)
            index.append('NaN')
            keeper.append(n_c)

        if zero == True:
            z_c = []
            for i in columns:
                zero_count = (sum(dfn[i] == '0')+sum(dfn[i] == 0))
                z_c.append(zero_count)
            index.append(0)
            keeper.append(z_c)

        if qmark == True:
            q_c = []
            for i in columns:
                q_mark_count = sum(dfn[i] == '?')
                q_c.append(q_mark_count)
            index.append('?')
            keeper.append(q_c)

        if missing == True:
            m_c = []
            for i in columns:
                missing_count = (sum(dfn[i] == '')+sum(dfn[i] == None))
                m_c.append(missing_count)
            index.append('Missing')
            keeper.append(m_c)
        if keeper == []:
            null_count = ('No null type requested.')
        else:
            null_count = pd.DataFrame(data=keeper, index=index, columns=columns)
        return (null_count)
               



if __name__ == '__main__':
    # Simple test for enlarger function
    # #print(enlarge(8))

    # # Test for train_validation_test_split function
    # raw_data = load_wine()
    # df = pd.DataFrame(data=raw_data['data'], columns=raw_data['feature_names'])
    # df['target'] = raw_data['target']

    # #X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split
    #     #(df, features=['ash', 'hue'], target='target')

    # # Test My_Data_Splitter
    # splitter = My_Data_Splitter(df=df, features=['ash', 'hue'], target='target')
    # X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_validation_test_split()
    # splitter.print_split_summary(X_train, X_val, X_test)


    # Setup DataFrame to test My_Ready_Frame methods

    data = ([[1, '', 1, 4, np.nan, 6, '0', 2],
            [2, 2, 1, 0, 1, 6, 6, 2],
            ['0', 'x', '? ', 'x', '  ', 0, 'x'],
            [np.nan, '?', '? ', 'c ', ' x', ' 0 ', ],
            [.4, .5, .35, ' ?', np.nan, .55, ],
        [5, .55, 0, .5, .2, .4, .0, .6]
            ])

    names = ['int_a', 'int_b', 'str_a', 'str_b', 'fl_a', 'fl_b']

    df = pd.DataFrame(data, index=names).T

#Simple Test for null_counts method
    z = My_Ready_Frame.null_counts(df, True, False, False, True)
    print(z)
    print('')

#Simple Test for clean_frame method
    q = My_Ready_Frame.clean_frame(df)
    print(q)

 #  breakpoint()