
# Data Cleaning and Preparation


```python
import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)
```

## Handling Missing Data


```python
string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data
string_data.isnull()
```


```python
string_data[0] = None
string_data.isnull()
```

### Filtering Out Missing Data

* `dropna` , `fillna`


```python
from numpy import nan as NA
data = pd.Series([1, NA, 3.5, NA, 7])
data.dropna()
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64




```python
data[data.notnull()]
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64




```python
# dropna by default drops any ROW containing NA 
data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
                     [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna()
data
cleaned
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# only drop rows of all NA 
data.dropna(how='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop columns 
data[4] = NA
print('data')
print(data)
print('\n \n Drop NA columns')
data.dropna(axis=1, how='all')
```

    data
         0    1    2   4
    0  1.0  6.5  3.0 NaN
    1  1.0  NaN  NaN NaN
    2  NaN  NaN  NaN NaN
    3  NaN  6.5  3.0 NaN
    
     
     Drop NA columns





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop rows of more than 2 NA
df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA
df.iloc[:2, 2] = NA
print('Original df: ')
print(df)
df.dropna()
df.dropna(thresh=2)
```

    Original df: 
              0         1         2
    0  0.476985       NaN       NaN
    1 -0.577087       NaN       NaN
    2  0.523772       NaN  1.343810
    3 -0.713544       NaN -2.370232
    4 -1.860761 -0.860757  0.560145
    5 -1.265934  0.119827 -1.063512
    6  0.332883 -2.359419 -0.199543





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.523772</td>
      <td>NaN</td>
      <td>1.343810</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.713544</td>
      <td>NaN</td>
      <td>-2.370232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.860761</td>
      <td>-0.860757</td>
      <td>0.560145</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.265934</td>
      <td>0.119827</td>
      <td>-1.063512</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.332883</td>
      <td>-2.359419</td>
      <td>-0.199543</td>
    </tr>
  </tbody>
</table>
</div>



### Filling In Missing Data


```python
df.fillna(0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.476985</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.577087</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.523772</td>
      <td>0.000000</td>
      <td>1.343810</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.713544</td>
      <td>0.000000</td>
      <td>-2.370232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.860761</td>
      <td>-0.860757</td>
      <td>0.560145</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.265934</td>
      <td>0.119827</td>
      <td>-1.063512</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.332883</td>
      <td>-2.359419</td>
      <td>-0.199543</td>
    </tr>
  </tbody>
</table>
</div>




```python
# use different fill values for each columns (key of the dict)
df.fillna({1: 0.5, 2: 0})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.476985</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.577087</td>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.523772</td>
      <td>0.500000</td>
      <td>1.343810</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.713544</td>
      <td>0.500000</td>
      <td>-2.370232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.860761</td>
      <td>-0.860757</td>
      <td>0.560145</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.265934</td>
      <td>0.119827</td>
      <td>-1.063512</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.332883</td>
      <td>-2.359419</td>
      <td>-0.199543</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fill na in-place 
_ = df.fillna(0, inplace=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.476985</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.577087</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.523772</td>
      <td>0.000000</td>
      <td>1.343810</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.713544</td>
      <td>0.000000</td>
      <td>-2.370232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.860761</td>
      <td>-0.860757</td>
      <td>0.560145</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.265934</td>
      <td>0.119827</td>
      <td>-1.063512</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.332883</td>
      <td>-2.359419</td>
      <td>-0.199543</td>
    </tr>
  </tbody>
</table>
</div>




```python
# interpolate the NA values by its forward(previous) value 
df = pd.DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = NA
df.iloc[4:, 2] = NA
print('original df:')
print(df)
df.fillna(method='ffill')
df.fillna(method='ffill', limit=2)
```

    original df:
              0         1         2
    0 -0.726213  0.222896  0.051316
    1 -1.157719  0.816707  0.433610
    2  1.010737       NaN -0.997518
    3  0.850591       NaN  0.912414
    4  0.188211       NaN       NaN
    5  2.003697       NaN       NaN





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.726213</td>
      <td>0.222896</td>
      <td>0.051316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.157719</td>
      <td>0.816707</td>
      <td>0.433610</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.010737</td>
      <td>0.816707</td>
      <td>-0.997518</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.850591</td>
      <td>0.816707</td>
      <td>0.912414</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.188211</td>
      <td>NaN</td>
      <td>0.912414</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.003697</td>
      <td>NaN</td>
      <td>0.912414</td>
    </tr>
  </tbody>
</table>
</div>




```python
# interpolate by mean
data = pd.Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())
```




    0    1.000000
    1    3.833333
    2    3.500000
    3    3.833333
    4    7.000000
    dtype: float64



## Data Transformation

### Removing Duplicates

* `duplicated` returns a boolean series indicating whether each row is a duplicate. 

* `drop_duplicates` returns a DataFrame where duplicated array is False.


```python
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
                     'k2': [1, 1, 2, 3, 3, 4, 4]})
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.duplicated()
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    dtype: bool




```python
data.drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop duplicates based on specific column 
data['v1'] = range(7)
data.drop_duplicates(['k1'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# keep the last observed duplicates
data.drop_duplicates(['k1', 'k2'], keep='last')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



### Transforming Data Using a Function or Mapping

* `map` performs element-wise transoformations and other data cleaning-related operations.


```python
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                              'Pastrami', 'corned beef', 'Bacon',
                              'pastrami', 'honey ham', 'nova lox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>food</th>
      <th>ounces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bacon</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pulled pork</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bacon</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pastrami</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>corned beef</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bacon</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pastrami</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>honey ham</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nova lox</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}
```


```python
lowercased = data['food'].str.lower()
lowercased
data['animal'] = lowercased.map(meat_to_animal)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>food</th>
      <th>ounces</th>
      <th>animal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bacon</td>
      <td>4.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pulled pork</td>
      <td>3.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bacon</td>
      <td>12.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pastrami</td>
      <td>6.0</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>corned beef</td>
      <td>7.5</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bacon</td>
      <td>8.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pastrami</td>
      <td>3.0</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>7</th>
      <td>honey ham</td>
      <td>5.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nova lox</td>
      <td>6.0</td>
      <td>salmon</td>
    </tr>
  </tbody>
</table>
</div>




```python
# wrap the mapping into a function
data['food'].map(lambda x: meat_to_animal[x.lower()])
```




    0       pig
    1       pig
    2       pig
    3       cow
    4       cow
    5       pig
    6       cow
    7       pig
    8    salmon
    Name: food, dtype: object



### Replacing Values

* `replace` provides simpler and more flexible way to modify a subset of values in an object. 


```python
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data
```




    0       1.0
    1    -999.0
    2       2.0
    3    -999.0
    4   -1000.0
    5       3.0
    dtype: float64




```python
data.replace(-999, np.nan)
```




    0       1.0
    1       NaN
    2       2.0
    3       NaN
    4   -1000.0
    5       3.0
    dtype: float64




```python
data.replace([-999, -1000], np.nan)
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    NaN
    5    3.0
    dtype: float64




```python
# pass substitutes with a list 
data.replace([-999, -1000], [np.nan, 0])
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    0.0
    5    3.0
    dtype: float64




```python
# use a dict to indicate the substitutes
data.replace({-999: np.nan, -1000: 0})
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    0.0
    5    3.0
    dtype: float64



### Renaming Axis Indexes


```python
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
```


```python
transform = lambda x: x[:4].upper()
data.index.map(transform)
```




    Index(['OHIO', 'COLO', 'NEW '], dtype='object')




```python
data.index = data.index.map(transform)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OHIO</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.rename(index=str.title, columns=str.upper)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ONE</th>
      <th>TWO</th>
      <th>THREE</th>
      <th>FOUR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colo</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>New</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.rename(index={'OHIO': 'INDIANA'},
            columns={'three': 'peekaboo'})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>peekaboo</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INDIANA</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.rename(index={'OHIO': 'INDIANA'}, inplace=True)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INDIANA</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



### Discretization and Binning

* `cut` bins the data based on the provided bins, can use `value_counts` to count each corresponding bin. 

* `qcut` computes bins based on sample quantiles. 


```python
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
```


```python
bins = [18, 25, 35, 60, 100]
cats = pd.cut(ages, bins)
cats
```




    [(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
    Length: 12
    Categories (4, interval[int64]): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]




```python
print('codes: ',cats.codes)
print('categories: ', cats.categories)
pd.value_counts(cats)
```

    codes:  [0 0 0 1 0 0 2 1 3 2 2 1]
    categories:  IntervalIndex([(18, 25], (25, 35], (35, 60], (60, 100]],
                  closed='right',
                  dtype='interval[int64]')





    (18, 25]     5
    (35, 60]     3
    (25, 35]     3
    (60, 100]    1
    dtype: int64




```python
# ( ] to [ )
pd.cut(ages, [18, 26, 36, 61, 100], right=False)
```




    [[18, 26), [18, 26), [18, 26), [26, 36), [18, 26), ..., [26, 36), [61, 100), [36, 61), [36, 61), [26, 36)]
    Length: 12
    Categories (4, interval[int64]): [[18, 26) < [26, 36) < [36, 61) < [61, 100)]




```python
# customized bin names 
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)
```




    [Youth, Youth, Youth, YoungAdult, Youth, ..., YoungAdult, Senior, MiddleAged, MiddleAged, YoungAdult]
    Length: 12
    Categories (4, object): [Youth < YoungAdult < MiddleAged < Senior]




```python
data = np.random.rand(20)
pd.cut(data, 4, precision=2)
```




    [(0.014, 0.25], (0.48, 0.71], (0.48, 0.71], (0.71, 0.94], (0.25, 0.48], ..., (0.71, 0.94], (0.71, 0.94], (0.014, 0.25], (0.25, 0.48], (0.48, 0.71]]
    Length: 20
    Categories (4, interval[float64]): [(0.014, 0.25] < (0.25, 0.48] < (0.48, 0.71] < (0.71, 0.94]]




```python
data = np.random.randn(1000)  # Normally distributed
cats = pd.qcut(data, 4)  # Cut into quartiles
cats
pd.value_counts(cats)
```




    (0.604, 3.928]                   250
    (-0.0453, 0.604]                 250
    (-0.684, -0.0453]                250
    (-2.9499999999999997, -0.684]    250
    dtype: int64




```python
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
```




    [(-1.187, -0.0453], (-1.187, -0.0453], (-0.0453, 1.289], (-1.187, -0.0453], (-0.0453, 1.289], ..., (-0.0453, 1.289], (-0.0453, 1.289], (-1.187, -0.0453], (-0.0453, 1.289], (-0.0453, 1.289]]
    Length: 1000
    Categories (4, interval[float64]): [(-2.9499999999999997, -1.187] < (-1.187, -0.0453] < (-0.0453, 1.289] < (1.289, 3.928]]



### Detecting and Filtering Outliers


```python
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.029706</td>
      <td>-0.013138</td>
      <td>-0.046007</td>
      <td>0.049045</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.009847</td>
      <td>0.997131</td>
      <td>0.998190</td>
      <td>1.001103</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.184377</td>
      <td>-3.745356</td>
      <td>-3.428254</td>
      <td>-3.645860</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.610007</td>
      <td>-0.700274</td>
      <td>-0.740152</td>
      <td>-0.599807</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.005308</td>
      <td>-0.033738</td>
      <td>-0.085190</td>
      <td>0.052073</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.695298</td>
      <td>0.692355</td>
      <td>0.625698</td>
      <td>0.750195</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.525865</td>
      <td>2.735527</td>
      <td>3.366626</td>
      <td>2.653656</td>
    </tr>
  </tbody>
</table>
</div>




```python
col = data[2]
col[np.abs(col) > 3]
```




    244   -3.428254
    621    3.366626
    Name: 2, dtype: float64




```python
# select all rows having a value exceeding 3 or -3, use any method
data[(np.abs(data) > 3).any(1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>-0.025907</td>
      <td>-3.399312</td>
      <td>-0.974657</td>
      <td>-0.685312</td>
    </tr>
    <tr>
      <th>46</th>
      <td>3.260383</td>
      <td>0.963301</td>
      <td>1.201206</td>
      <td>-1.852001</td>
    </tr>
    <tr>
      <th>122</th>
      <td>-0.196713</td>
      <td>-3.745356</td>
      <td>-1.520113</td>
      <td>-0.346839</td>
    </tr>
    <tr>
      <th>221</th>
      <td>-3.056990</td>
      <td>1.918403</td>
      <td>-0.578828</td>
      <td>1.847446</td>
    </tr>
    <tr>
      <th>244</th>
      <td>0.326045</td>
      <td>0.425384</td>
      <td>-3.428254</td>
      <td>-0.296336</td>
    </tr>
    <tr>
      <th>308</th>
      <td>-3.184377</td>
      <td>1.369891</td>
      <td>-1.074833</td>
      <td>-0.089937</td>
    </tr>
    <tr>
      <th>529</th>
      <td>0.208011</td>
      <td>-0.150923</td>
      <td>-0.362528</td>
      <td>-3.548824</td>
    </tr>
    <tr>
      <th>621</th>
      <td>0.193299</td>
      <td>1.397822</td>
      <td>3.366626</td>
      <td>-2.372214</td>
    </tr>
    <tr>
      <th>768</th>
      <td>3.525865</td>
      <td>0.283070</td>
      <td>0.544635</td>
      <td>0.462204</td>
    </tr>
    <tr>
      <th>788</th>
      <td>-0.450721</td>
      <td>-0.080332</td>
      <td>0.599947</td>
      <td>-3.645860</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[np.abs(data) > 3] = np.sign(data) * 3
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.029161</td>
      <td>-0.011993</td>
      <td>-0.045945</td>
      <td>0.050240</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.006613</td>
      <td>0.993332</td>
      <td>0.995639</td>
      <td>0.997092</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.000000</td>
      <td>-3.000000</td>
      <td>-3.000000</td>
      <td>-3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.610007</td>
      <td>-0.700274</td>
      <td>-0.740152</td>
      <td>-0.599807</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.005308</td>
      <td>-0.033738</td>
      <td>-0.085190</td>
      <td>0.052073</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.695298</td>
      <td>0.692355</td>
      <td>0.625698</td>
      <td>0.750195</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>2.735527</td>
      <td>3.000000</td>
      <td>2.653656</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.sign(data).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### Permutation and Random Sampling

* `np.random.permutation` 


```python
df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
sampler
```




    array([0, 4, 3, 1, 2])




```python
print('df: ')
print(df)
df.take(sampler)
```

    df: 
        0   1   2   3
    0   0   1   2   3
    1   4   5   6   7
    2   8   9  10  11
    3  12  13  14  15
    4  16  17  18  19





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# a random subset without replacement 
df.sample(n=3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# generate sample with replacement 
choices = pd.Series([5, 7, -1, 6, 4])
draws = choices.sample(n=10, replace=True)
draws
```




    4    4
    1    7
    3    6
    2   -1
    3    6
    2   -1
    3    6
    4    4
    1    7
    4    4
    dtype: int64



### Computing Indicator/Dummy Variables

* `pd.get_dummies`

* Use `get_dummies` together with `cut` for discretize continuous variables.


```python
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                   'data1': range(6)})
print(df)
pd.get_dummies(df['key'])
```

      key  data1
    0   b      0
    1   b      1
    2   a      2
    3   c      3
    4   a      4
    5   b      5





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data1</th>
      <th>key_a</th>
      <th>key_b</th>
      <th>key_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_csv('datasets/movielens/movies.dat', sep='::',
                       header=None, names=mnames, engine = 'python')
movies[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children's|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Heat (1995)</td>
      <td>Action|Crime|Thriller</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Sabrina (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Tom and Huck (1995)</td>
      <td>Adventure|Children's</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Sudden Death (1995)</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>GoldenEye (1995)</td>
      <td>Action|Adventure|Thriller</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_genres = []
for x in movies.genres:
    all_genres.extend(x.split('|'))
genres = pd.unique(all_genres)
```


```python
genres
```




    array(['Animation', "Children's", 'Comedy', 'Adventure', 'Fantasy',
           'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror',
           'Sci-Fi', 'Documentary', 'War', 'Musical', 'Mystery', 'Film-Noir',
           'Western'], dtype=object)




```python
zero_matrix = np.zeros((len(movies), len(genres)))
dummies = pd.DataFrame(zero_matrix, columns=genres)
```


```python
gen = movies.genres[0]
gen.split('|')
dummies.columns.get_indexer(gen.split('|'))
```




    array([0, 1, 2])




```python
for i, gen in enumerate(movies.genres):
    indices = dummies.columns.get_indexer(gen.split('|'))
    dummies.iloc[i, indices] = 1
```


```python
movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.iloc[0]
```




    movie_id                                       1
    title                           Toy Story (1995)
    genres               Animation|Children's|Comedy
    Genre_Animation                                1
    Genre_Children's                               1
    Genre_Comedy                                   1
    Genre_Adventure                                0
    Genre_Fantasy                                  0
    Genre_Romance                                  0
    Genre_Drama                                    0
                                    ...             
    Genre_Crime                                    0
    Genre_Thriller                                 0
    Genre_Horror                                   0
    Genre_Sci-Fi                                   0
    Genre_Documentary                              0
    Genre_War                                      0
    Genre_Musical                                  0
    Genre_Mystery                                  0
    Genre_Film-Noir                                0
    Genre_Western                                  0
    Name: 0, Length: 21, dtype: object




```python
np.random.seed(12345)
values = np.random.rand(10)
values
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))
```




       (0.0, 0.2]  (0.2, 0.4]  (0.4, 0.6]  (0.6, 0.8]  (0.8, 1.0]
    0           0           0           0           0           1
    1           0           1           0           0           0
    2           1           0           0           0           0
    3           0           1           0           0           0
    4           0           0           1           0           0
    5           0           0           1           0           0
    6           0           0           0           0           1
    7           0           0           0           1           0
    8           0           0           0           1           0
    9           0           0           0           1           0



## String Manipulation

### String Object Methods





```python
val = 'a,b,  guido'
val.split(',')
```




    ['a', 'b', '  guido']




```python
pieces = [x.strip() for x in val.split(',')]
pieces
```




    ['a', 'b', 'guido']




```python
first, second, third = pieces
first + '::' + second + '::' + third
```




    'a::b::guido'




```python
# a better way to join a list or tuple 
'::'.join(pieces)
```




    'a::b::guido'




```python
# index raises error if string is not found, `find` returns -1 if not found 
'guido' in val
val.index(',')
val.find(':')
```




    -1




```python
val.index(':')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-86-2c016e7367ac> in <module>
    ----> 1 val.index(':')
    

    ValueError: substring not found



```python
val.count(',')
```




    2




```python
val.replace(',', '::')
val.replace(',', '')
```




    'ab  guido'



### Regular Expressions

* Pattern matching, substitution, splitting

* `re` module 

* `\s+` is used to describe one or more whitespace characters 


```python
import re
text = "foo    bar\t baz  \tqux"
re.split('\s+', text)
```




    ['foo', 'bar', 'baz', 'qux']




```python
regex = re.compile('\s+')
regex.split(text)
```




    ['foo', 'bar', 'baz', 'qux']




```python
regex.findall(text)
```




    ['    ', '\t ', '  \t']




```python
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

# re.IGNORECASE makes the regex case-insensitive
regex = re.compile(pattern, flags=re.IGNORECASE)
```


```python
regex.findall(text)
```




    ['dave@google.com', 'steve@gmail.com', 'rob@gmail.com', 'ryan@yahoo.com']




```python
m = regex.search(text)
m
text[m.start():m.end()]
```




    'dave@google.com'




```python
# match will only match if the pattern occurs at the start of the string 
print(regex.match(text))
```

    None



```python
print(regex.sub('REDACTED', text))
```

    Dave REDACTED
    Steve REDACTED
    Rob REDACTED
    Ryan REDACTED
    



```python
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
```


```python
# groups returns a tupe of the pattern components in match
m = regex.match('wesm@bright.net')
m.groups()
```




    ('wesm', 'bright', 'net')




```python
regex.findall(text)
```




    [('dave', 'google', 'com'),
     ('steve', 'gmail', 'com'),
     ('rob', 'gmail', 'com'),
     ('ryan', 'yahoo', 'com')]




```python
print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))
```

    Dave Username: dave, Domain: google, Suffix: com
    Steve Username: steve, Domain: gmail, Suffix: com
    Rob Username: rob, Domain: gmail, Suffix: com
    Ryan Username: ryan, Domain: yahoo, Suffix: com
    


### Vectorized String Functions in pandas

* `str` attribute in Series allows string operations skipping the NA values.


```python
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = pd.Series(data)
data
data.isnull()
```




    Dave     False
    Steve    False
    Rob      False
    Wes       True
    dtype: bool




```python
data.str.contains('gmail')
```




    Dave     False
    Steve     True
    Rob       True
    Wes        NaN
    dtype: object




```python
pattern
data.str.findall(pattern, flags=re.IGNORECASE)
```




    Dave     [(dave, google, com)]
    Steve    [(steve, gmail, com)]
    Rob        [(rob, gmail, com)]
    Wes                        NaN
    dtype: object




```python
matches = data.str.match(pattern, flags=re.IGNORECASE)
matches
```




    Dave     True
    Steve    True
    Rob      True
    Wes       NaN
    dtype: object




```python
matches.str.get(1)
matches.str[0]
```




    Dave    NaN
    Steve   NaN
    Rob     NaN
    Wes     NaN
    dtype: float64




```python
data.str[:5]
```




    Dave     dave@
    Steve    steve
    Rob      rob@g
    Wes        NaN
    dtype: object




```python
pd.options.display.max_rows = PREVIOUS_MAX_ROWS
```
