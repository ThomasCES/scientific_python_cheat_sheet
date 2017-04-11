Scientific Python Cheatsheet
============================

# Pure Python
### Types

```python
a = 2           # integer
b = 5.0         # float
c = 8.3e5       # = 8.3 * 10**5
d = 1.5 + 0.5j  # complex
e = True        # boolean (or False)
f = 'word' (or "word")      # string
```
### Operators

```python
3 + 2             # addition
3 / 2             # integer (python2) or float (python3) division
3 // 2            # integer division (division euclidienne)
3 * 2             # multiplication
3 ** 2            # exponent
3 % 2             # remainder (reste de la division euclidienne)
abs(a)            # absolute value
a += 1 (*=, /=)   # change and assign (a = a + 1) 
1 == 1            # equal, ask question (= True)
2 > 1             # larger (= True)
2 < 1             # smaller(= False)
1 != 2            # not equal
1 != 2 and 2 < 3  # logical AND
1 != 2 or 2 < 3   # logical OR
a is b            # test if objects point to the same memory (id)
```

### Strings

```python
a = 'red'                      # assignment
char = a[2]                    # access individual characters (='d')
'red ' + 'blue'                # string concatenation ( ='redblue')
'1, 2, three'.split(',')       # split string into list
'1, 2, three'.replace(',','.') # replace character by other one (',' by '.' in the example)
'.'.join(['1', '2', 'three'])  # concatenate list into string
```

### Lists

```python
a = [] (or a = list())             # create empty list
a = ['red', 'blue', 'green']       # manually initialization
b = list(range(5))                 # initialize from iteratable
c = [nu**2 for nu in b]            # list comprehension
d = [nu**2 for nu in b if nu < 3]  # conditioned list comprehension
e = a[0]                           # access element
f = a[1:2]                         # access a slice of the list
g = a[-1]                          # access last element
h = ['re', 'bl'] + ['gr']          # list concatenation
i = ['re'] * 5                     # repeat a list
['do', 're', 'mi'].index('re')     # returns index of 're'
a.append('yellow')                 # add new element to end of list
a.insert(1, 'yellow')              # insert element in specified position
're' in ['do', 're', 'mi']         # true if 're' in list
'fa' not in ['do', 're', 'mi']     # true if 'fa' not in list
sorted([3, 2, 1])                  # returns sorted list (work with any iterable object)
a.remove('red')                     # remove item from list
```

### Dictionaries

```python
a = {'red': 'rouge', 'blue': 'bleu'}         # dictionary
b = a['red']                                 # call item
'red' in a                                   # true if dictionary a contains key 'red'
c = [value for key, value in a.items()]      # loop through contents
a.update({'green': 'vert', 'brown': 'brun'}) # update dictionary by data from another one
a.keys()                                     # get list of keys
a.values()                                   # get list of values
a.items()                                    # get list of key-value pairs
del a['red']                                 # delete key and the associated value
```

### Control Flow : if, for, while

```python
# if/elif/else
a, b = 1, 2
if a + b == 3:
    print('True')
elif a + b == 1:
    print('False')
else:
    print('?')

# for
a = ['red', 'blue', 'green']
for i in a:
    print(i)
    
# for (bis) i = 0 to 9
for i in range(0,10):
    print(i**2)
    
# while
number = 1
while number < 10:
    print(number)
    number += 1

# break
number = 1
while True:
    print(number)
    number += 1
    if number > 10:
        break
```
### operating system interfaces (`import os as os`)
see https://docs.python.org/3.6/library/os.html#
```python
folder_name = os.getcwd()
file_name = os.listdir(folder_name)
os.chdir(another_folder)
```
# NumPy (`import numpy as np`)
### array initialization

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6]) # vector, direct initialization
x = np.array([[2, 7, 1], [8, 2, 8]) # matrix, direct initialization
x = np.array([a, a, a])             # matrix, 3 rows
b = np.zeros(8)                     # vector initialized with 8 zeros
c = np.ones((3,3))                  # 3 x 3 integer matrix with ones
d = np.eye(200)                     # ones on the diagonal
e = np.linspace(0., 10., 100)       # 100 points from 0 to 10
f = np.arange(0, 100, 2)            # points from 0 to <100 with step 2
g = np.logspace(-5, 2, 100)         # 100 log-spaced from 1e-5 -> 1e2
h = np.copy(a)                      # copy array to new memory
```

### indexing

```python
a[:3] = 0                 # set the first three indices to zero
a[2:5] = 1                # set indices 2-4 to 1
a[:-3] = 2                # set all but last three elements to 2
a[start:stop:step]        # general form of indexing/slicing
a[None, :]                # transform to column vector
a[[1, 1, 3, -1]]           # return array with values of the indices
a[a < 2]                  # values with elementwise condition
a[a > 2] = 0              # set values equal to 0 under condition   
```

### array properties and operations

```python
a.shape                     # a tuple with the lengths of each axis
len(a)                      # length of axis 0 (ie. number of row)
a.ndim                      # number of dimensions (axes)
a.sort(axis=1)              # sort array along axis
a.flatten()                 # collapse array to one dimension
a = a.reshape(2, 4)         # transform to 2 x 4 matrix
a.T                         # return transposed view
a.tolist()                  # convert (possibly multidimensional) array to list
np.argmax(a, axis=1)        # return index of maximum along a given axis
np.cumsum(a)                # return cumulative sum
np.any(a)                   # True if any element is True
np.all(a)                   # True if all elements are True
np.argsort(a, axis=1)       # return sorted index array along axis
np.where(cond)              # return indices where cond is True
```

### boolean arrays

```python
a < 2                         # returns array with boolean values
(a < 2) & (b == 0)            # elementwise logical and
(a < 2) | (b != 0)            # elementwise logical or
```

### math functions

```python
a * 5              # multiplication with scalar
a + 5              # addition with scalar
a + b              # addition with array b
a / b              # division with b (np.NaN for division by zero)
np.exp(a)          # exponential (complex and real)
np.sin(a)          # sine
np.cos(a)          # cosine
np.arctan2(a, b)   # ~arctan(a/b)
np.arcsin(a)       # arcsin
np.radians(a)      # degrees to radians
np.degrees(a)      # radians to degrees
np.var(a)          # variance of array
np.std(a, axis=1)  # standard deviation
np.dot(a, b)       # matrix product (inner product: a_mi b_in)
np.sum(a, axis=1)  # sum over axis 1
np.abs(a)          # return absolute values
np.round(a)        # rounds to neares int
```

### linear algebra/ matrix math

```python
evals, evecs = np.linalg.eig(a)   # find eigenvalues and eigenvectors
coef = np.polyfit(x,y,2)          # return values of polynomial factors (2nd order in the example)
coef = np.corrcoef(x,y)           # return correlation coefficients (R and not R² like Excel)
```

### reading/ writing files

```python
np.savetxt('data.txt', x , fmt='%1.4e', delimiter=';')          # write 'x' in ascii data
data = np.loadtxt('data.txt', skiprows=0, delimiter=';')   # read ascii data from file
```

### interpolation, integration, optimization

```python
np.trapz(a, x=x, axis=1)  # integrate along axis 1
np.interp(x, xp, yp)      # interpolate function xp, yp at points x
np.linalg.lstsq(a, b)     # solve a x = b in least square sense
np.linalg.det(a)          # compute the determinant of a (n,n)
np.linalg.eigvals(a)      # compute the eigenvalues of a
```

### random variables

```python
np.random.normal(loc=0, scale=2, size=100)  # 100 normal distributed
xx = np.random.rand(100)                         # 100 random numbers in [0, 1]
yy = np.random.uniform(1, 42, 100)               # 100 random numbers in [1, 42]
zz = np.random.randint(1, 42, [100,  100])       # 100 random integers in [1, 42]
np.random.choice([0, 1], 100, p=[0.1, 0.9])   # 100 random numbers choose in a list with p probability
```

# Functions

```python
# Function to compute simple polynom
import numpy as np
def myfunc(x, a, b):
    y = a * x + b
    z = a * x**2 + b * x
    return y, z

x = np.arange(0,100,1)
a = 0.33
b = -45
(y, z) = myfunc(x, a, b)  #call "myfunc"
```
# Matplotlib (`import matplotlib.pyplot as plt`)
### figures and axes

```python
plt.figure(1)  # initialize figure
fig, axes = plt.subplots(5, 2, figsize=(5, 5))  # figure with 10 plots and 5 x 2 axes
plt.savefig('out.png', bbox_inches='tight')     # save png image
```

### figures and axes properties

```python
plt.title('title')            # figure title
plt.xlabel('xbla')            # set xlabel
plt.ylabel('ybla')            # set ylabel
plt.xlim(0, 2)                # sets x limits
plt.ylim(0, 4)                # sets y limits
plt.legend(['case A','case B'], loc='best')    # show legend
```

### plotting 
see http://matplotlib.org/gallery.html

```python
plt.plot(xx, yy, '-o', c='red', lw=2)            # plots a line
plt.scatter(xx , yy, s=20, c = 'black')          # scatter plot
plt.pcolormesh(xx, yy, zz, shading='gouraud')    # colormesh
plt.contour(xx, yy, zz, cmap='jet')              # contour lines
plt.contourf(xx, yy, zz, vmin=2, vmax=4)         # filled contours
plt.hist(xx, bins=50)                            # histogram
plt.imshow(matrix, origin='lower', extent=(x1, x2, y1, y2),
        interpolation='bilinear', aspect='auto') # image (carpet plot, heat map)
ax.specgram(y, FS=0.1, noverlap=128,
            scale='linear')                      # spectrogram
ax.text(x, y, string, fontsize=12, color='m')    # write text
```

## Pandas (`import pandas as pd`)

### DataFrame (DF)
```python
df = pd.DataFrame()                # create an empty DataFrame
df['A'] = [0, 1, 'lundi', 3, 4]    # store list in 'A' column
df['B'] = np.arange(5)             # store array in 'B' column
print(df[:2])                      # print first 2 lines of the DF
a = df.values                      # get data out of DF
a = df['A'].values                 # get the 'A' column out of DF
a = df.iloc[2,3]                   # get element (indexe [2,3]) out of DF
cols = df.columns                  # get list of columns names
df2 = df.dropna(axis=1,how='all')  # delete "empty" cell of DF
df2 = df.fillna(value = 5)         # replace "empty cell by  the number 5 in DF
df.isnull()                        # detect empty cells in the DF
df.isin([1,2])                     # boolean showing if each element in the DF is contained in the list
df2 = df[df['A'] == 0]             # create a new DF by selecting raws under condition
a = df['B'].mean()                 # compute de mean value of the column 'A' (work with sum(), min(), max()...)
dfI = df.interpolate(method='time')     # data interpolation (gap completion)
dfR = df.resample(rule = '30Min').mean() # change time step to 30 min with the mean methode
df.index = pd.to_datetime(TimeVector)    # use TimeVector to creat new index in the DF
df.plot()                          # use matplotlib to plot the DF (many options!)
```

### read/write data
```python
df = pd.read_csv("filename.csv", sep=',', skiprows='0')   # read and load CSV (or .txt) file in a DF
df = pd.read_excel("filename.xls") # read and load excel sheet in a DF
df.to_excel("filename.xls")        # save DF in Excel file
df.to_csv("filename.csv")          # save DF in text file
```
