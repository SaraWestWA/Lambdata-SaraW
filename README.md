
# Lambdata-SaraW
A collection of data science utility functions.

## Installation

TO DO

## Usage

TO DO

## Usage

```
from_mylambdata.ds_utilities import enlarge

print(enlarge(9))
```

```
pip install -i https://test.pypi.org/simple/ Lambdata-SaraW==0.0.8

from mylambdata.ds_utilities import My_Ready_Frame

q = My_Ready_Frame.null_counts(df, True, True, True, True)
print(q)

returns new dataframe of null values per column
```

```
from mylambdata.ds_utilities import My_Ready_Frame

z = My_Ready_Frame.clean_frame(df)
print(z)

returns dataframe with spaces removed and missing values
replaced with np.nan, column dtypes set to int64 if possible
```

