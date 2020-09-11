import pandas as pd 
import numpy as np
import warnings


# class Null_Counts():
#     def __init__(self, df):
#         self.df = df

#         """
#         Param is a dataframe, can have both catagorical and numeric data

#         Function returns a dataframe of counts for  null values and 0.
#         """
#         # surpress warning comparing pd objects to np.nan
#         warnings.simplefilter(action='ignore', category=FutureWarning)
#         df = df.applymap(lambda x: x.strip() if type(x) == str else x)
#         columns = list(df.columns)
#         n_c = []
#         z_c = []
#         q_c = []
#         m_c = []
#         for i in columns:
#             nan_count = df[i].isnull().sum()
#             zero_count = (sum(df[i] == '0')+sum(df[i] == 0))
#             q_mark_count = sum(df[i] == '?')
#             missing_count = (sum(df[i] == '')+sum(df[i] == None))
#             n_c.append(nan_count)
#             z_c.append(zero_count)
#             q_c.append(q_mark_count)
#             m_c.append(missing_count)
#         null_count = pd.DataFrame(data=(n_c, z_c, q_c, m_c),
#                                 index=['NaN', 0, '?', 'Missing'],
#                                 columns=columns)
#         return (null_count)


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

    data = ([[1, '', 1, 4, np.nan, 6, '0', 2],
            [2, 2, 1, 0, 1, 6, 6, 2],
            ['0', 'x', '? ', 'x', '  ', 0, 'x'],
            [np.nan, '?', '? ', 'c ', ' x', ' 0 ', ],
            [.4, .5, .35, ' ?', np.nan, .55, ],
        [5, .55, 0, .5, .2, .4, .0, .6]
            ])

    names = ['int_a', 'int_b', 'str_a', 'str_b', 'fl_a', 'fl_b']

    df = pd.DataFrame(data, index=names).T
    z = My_Ready_Frame.clean_frame(df)

    print(z)

