from mylambdata.ds_utilities import enlarge
from mylambdata.ds_utilities import null_counts
import numpy as np
import pandas as pd
from mylambdata.ds_utilities import clean_frame

data = ([[1, '', 1, 4, np.nan, 6, '0', 2],
            [2, 2, 1, 0, 1, 6, 6, 2],
            ['0', 'x', '? ', 'x', '  ', 0, 'x'],
            [np.nan, '?', '? ', 'c ', ' x', ' 0 ', ],
            [.4, .5, .35, ' ?', np.nan, .55, ],
            [5, .55, 0, .5, .2, .4, .0, .6]
             ])

names = ['int_a', 'int_b', 'str_a', 'str_b', 'fl_a', 'fl_b']

df = pd.DataFrame(data, index=names).T

nulls = null_counts(df)
print(nulls)

clean_frame (df)
print(df)