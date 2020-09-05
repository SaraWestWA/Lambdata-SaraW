import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_wine
# from pdb import set_trace as breakpoint


def enlarge(n):
    """
    Parameter n is a #
    Function will muliply by 100
    """
    return n*100





"""
null_counts examines a dataframe and returns another dataframe with
counts for various null values as well as zeros.
"""

def null_counts (df):
  # surpress warning comparing pd objects to np.nan
  # warnings.simplefilter(action='ignore', category=FutureWarning)
  df = df.applymap(lambda x: x.strip() if type(x)==str else x)
  columns = list(df.columns)
  n_c = []
  z_c = []
  q_c = []
  m_c = []
  for i in columns:
    nan_count = df[i].isnull().sum()
    zero_count = (sum(df[i]=='0')+sum(df[i]==0))
    q_mark_count = sum(df[i]=='?')
    missing_count = (sum(df[i]=='')+sum(df[i]==None))
    n_c.append(nan_count)
    z_c.append(zero_count)
    q_c.append(q_mark_count)
    m_c.append(missing_count)
  null_count = pd.DataFrame(data=(n_c,z_c,q_c,m_c), index = ['NaN',0, '?', 'Missing'], columns = columns)
  return (null_count)

"""
clean_frame removes leading & trailing spaces, replaces '?', '', and empty cells
with NaN and changes dtype to float64 if possible.
"""

def clean_frame(df):
  df = df.applymap(lambda x: x.strip() if type(x)==str else x)
  df = df.applymap(lambda x: np.nan if type(x)==str and x=='' or x== None or x=='?' else x)
  df = df.apply(pd.to_numeric,errors='ignore') 
  return (df)

if __name__ == '__main__':
    print(enlarge(8))
    