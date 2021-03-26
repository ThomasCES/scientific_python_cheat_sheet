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
f = 'word'      # string (or "word")
```
### Operators

```python
x = 3 + 2             # addition
x = 3 / 2             # integer (python2) or float (python3) division
x = 3 // 2            # integer division (division euclidienne)
x = 3 * 2             # multiplication
x = 3**2              # exponent
x = 3 % 2             # remainder (reste de la division euclidienne)
a += 1                # change and assign : a = a + 1 (*=, /=)
x = 1 == 1            # equal, ask question (= True)
x = 2 > 1             # larger (= True)
x = 2 <= 1            # smaller or equal (= False)
x = 1 != 2            # not equal
x = 1 != 2 and 2 < 3  # logical AND
x = 1 != 2 or 2 < 3   # logical OR
```

### Strings

```python
a = 'orange'                   # assignment
b = a[4]                       # access individual characters (='g')
x = 'red ' + 'blue'                # string concatenation ( ='redblue')
x = 'one,two,three'.split(',')     # split string into list
x = 'one,two,three'.replace(',','.') # replace character by other one (',' by '.' in the example)
x = ','.join(['one', 'two', 'three'])  # concatenate list into string
```

### Lists

```python
a = []                             # create empty list (or a = list())
a = ['red', 'blue', 'green']       # manually initialization
e = a[0]                           # access first element
f = a[0:2]                         # access a slice of the list
g = a[-1]                          # access last element
h = ['re', 'bl'] + ['gr']          # list concatenation
i = ['re'] * 5                     # repeat a list five time
x = ['do', 're', 'mi'].index('re')     # returns index of 're'
a.append('yellow')                 # add new element to end of list
a.insert(1, 'yellow')              # insert element in specified position
're' in ['do', 're', 'mi']         # true if 're' in list
'fa' not in ['do', 're', 'mi']     # true if 'fa' not in list
x = sorted([3, 2, 1])                  # returns sorted list
a.remove('red')                    # remove item from list
x = len(a)                             # length of the list (or any python object)
b = list(range(5))                 # initialize from iteratable
c = [nu**2 for nu in b]            # list comprehension
d = [nu**2 for nu in b if nu < 3]  # conditioned list comprehension
```

### Dictionaries

```python
a = {'red': 'rouge', 'blue': 'bleu'}    # dictionary
b = a['red']              # call item
'red' in a                # true if dictionary a contains key 'red'
a.keys()                  # get list of keys
a.values()                # get list of values
a.items()                 # get list of key-value pairs
del a['red']              # delete key and the associated value
for k, v in a.items():    # loop through contents and print values
    print(v)
```

### Control Flow : if, for, while

```python
# for loop
a = ['red', 'blue', 'green']
for i in a:
    print(i)
    
# for loop (bis)
resu = []   #create an empty list
for i in range(0,10):
    resu.append(i)  #store "i" in the list each loop
print(resu)
    
# while / if / else
number = 1
resu2 = []
resu3 = []
while number < 100: # (same as "for number in range(100):")
    if number % 2 == 0:
        resu2.append(number)
    if number % 3 == 0:
        resu3.append(number)
    else:
        pass
    number = number+1
    
```
# operating system interfaces (`import os as os`)
see https://docs.python.org/3.6/library/os.html#
```python
folder_name = os.getcwd()           # get working directory name
file_name = os.listdir(folder_name) # list of all the files in folder
os.chdir(another_folder)            # change working directory
```
# NumPy (`import numpy as np`)
### array initialization

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6]) # vector, direct initialization
x = np.array([[2, 7, 1], [8, 2, 8]]) # matrix, direct initialization
x1 = np.array([a, a, a])    # matrix, 3 rows
b = np.zeros(8)            # vector initialized with 8 zeros
c = np.ones([3,3])         # 3 x 3 integer matrix with ones
d = np.eye(200)            # ones on the diagonal
e = np.linspace(0, 10, 100)  # 100 points from 0 to 10
f = np.arange(0, 100, 2)     # points from 0 to <100 with step 2
g = np.logspace(-5, 2, 100)  # 100 log-spaced from 1e-5 -> 1e2
h = np.copy(a)               # copy array to new memory
```

### indexing
#### a[start:stop:step]        # general form of indexing/slicing
```python
z = a[0]          # store the first element of a in "z"
z = a[0:4]        # store the first four elemnts of "a" in ""z
z = a[-1]         # store the last element of "a" in "z"
z = a[[1, 1, 3, -1]]  # store elements 1,1,3,-1 of "a" in "z"
a[0:3] = 0        # set the first three indices to zero (or a[:3]=0)
a[2] = 1          # set indice 3 to 1 (or a[2:3] = 1)
b[:-3] = 2        # set all but last three elements to 2
z = a[a > 2]      #  store values of "a" in "z" under condition 
a[a > 2] = 0      # set values equal to 0 under condition on "a"
a[b > 0] = -99    # set values equal to -99 under condition on "b"
```

### array properties and operations

```python
x = a.shape                     # size of the matrixe (or np.shape(a))
x = a.sort()                    # sort array along (or a = np.sort(a))
x = x1.flatten()                # collapse array to one dimension
x = a.reshape(2, 4)             # transform to 2 x 4 matrix
x = a.T                         # return transposed view
x = a.tolist()                  # convert (possibly multidimensional) array to list
x = np.argmax(a)                # return index of maximum along a given axis
x = np.cumsum(a)                # return cumulative sum
x = np.any([True, False, True]) # True if any element is True
x = np.all([True, False, True]) # True if all elements are True
x = np.argsort(a)               # return sorted index array along axis
x = np.where(x1 < 2, x1, x1*2)  # x = x1 if condition is True and x1*2 if False
x = np.isin(a,b)                # return true if elements of 'a' are in 'b'
```

### boolean arrays

```python
x = a < 2                 # returns array with boolean values
x = (a < 2) & (b == 0)    # elementwise logical and
x = (a < 2) | (b != 0)    # elementwise logical or
```

### math functions

```python
x = a * 5              # multiplication with scalar
x = a + 5              # addition with scalar
x = a + b              # addition with array b
x = a / b              # division with b (np.inf for division by zero)
x = np.exp(a)          # exponential (complex and real)
x = np.sin(a)          # sine (np.cos, np.arcsin, ...)
x = np.var(a)          # variance of array
x = np.std(a)          # standard deviation
x = np.dot(a, a)       # matrix product (inner product: a_mi b_in)
x = np.sum(a)          # sum of all numbers in a (np.mean, np.min, np.max, ...)
x = np.sum(x1, axis=1) # sum over axis 1 in x (try with axis = 0)       
x = np.abs(a)          # return absolute values
x = np.round(a)        # rounds to neares int
```

### linear algebra/ matrix math

```python
evals, evecs = np.linalg.eig(a)   # find eigenvalues and eigenvectors
coef = np.polyfit(x,y,2)          # return values of polynomial factors (2nd order in the example)
coef = np.corrcoef(y_ref,y_model) # return correlation coefficients (R and not R² like Excel)
```

### reading/ writing files

```python
np.savetxt('data.txt', x , fmt='%1.4e', delimiter=';')     # write 'x' in ascii data
data = np.loadtxt('data.txt', skiprows=0, delimiter=';')   # load ascii data from file
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
xx = np.random.rand(100)                      # 100 random numbers in [0, 1]
yy = np.random.uniform(1, 42, 100)            # 100 random numbers in [1, 42]
zz = np.random.randint(1, 42, [100,  100])    # 100 random integers in [1, 42]
x = np.random.choice([0, 1], 100, p=[0.1, 0.9])   # 100 random numbers choose in a list with p probability
x = np.random.normal(loc=0, scale=2, size=100)    # 100 normal distributed
```

# Functions

```python
# Function to compute polynoms
import numpy as np
def myfunc(x, a, b):
    y = a * x + b
    z = a * x**2 + b * x
    return y, z
# fonction utilization
x = np.arange(0,100,1)
a = 0.33
b = -45
y, z = myfunc(x, a, b)  #call "myfunc"
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
plt.title('title')        # figure title
plt.xlabel('time (°C)')   # set xlabel
plt.ylabel('power (kW)')  # set ylabel
plt.xlim(0, 2)            # sets x limits
plt.ylim(0, 4)            # sets y limits
plt.legend(['case A','case B'], loc='best')    # show legend
```

### plotting 
see http://matplotlib.org/gallery.html

```python
x = np.arange(0,100,0.1)
y = np.sin(x)
plt.figure(1)
plt.plot(x, y)
plt.figure(2)
plt.plot(x, y, '-o', c='red', lw=2)              # plots a line
plt.figure(3)
plt.bar(range(7),[5,5,8,6,9,2,3])                # plot bars
plt.figure(4)
plt.hist(y, bins=20)                             # histogram
plt.scatter(xx , yy, s=20, c = 'black')          # scatter plot
plt.pcolormesh(xx, yy, zz, shading='gouraud')    # colormesh
plt.contour(xx, yy, zz, cmap='jet')              # contour lines
plt.boxplot(matrix, showfliers=True)             # distribution of data based on the first quartile, median, third quartile 
plt.imshow(matrix, origin='lower', extent=(x1, x2, y1, y2),
        interpolation='bilinear', aspect='auto') # image (carpet plot, heat map)
ax.specgram(y, FS=0.1, noverlap=128,
            scale='linear')                      # spectrogram
```

# Pandas (`import pandas as pd`)

### DataFrame (DF)
```python
df = pd.DataFrame()                # create an empty DataFrame
df['A'] = [0, 1, 'lundi', 3, 4]    # store list in 'A' column
df['B'] = np.arange(5)             # store array in 'B' column
df['C'] = [0, 1, np.nan, 3, 4]     # store list in 'C' column with an empty cell (ei. np.nan)
print(df[:2])                      # print first 2 lines of the DF
a = df.values                      # get data out of DF
a = df['A'].values                 # get the 'A' column out of DF
a = df.iloc[2,1]                   # get element (index [2,1]) out of DF
a = df.iloc[0:3,1:3]               # get range of elements out of DF
cols = df.columns                  # get list of columns names
df.isin([1,2])                     # boolean showing if each element in the DF is contained in the list
df2 = df[df['B'] > 2]             # create a new DF by selecting raws under condition
a = df['B'].mean()                 # compute de mean value of the column 'B' (work with sum(), min(), max()...)
df2 = df.dropna(axis=1)            # delete "empty" cell of DF
df2 = df.drop(columns = ['A'])     # delete column 'A' of DF
df2 = df.fillna(value = 99)        # replace "empty" cell by 99 in DF
df.isnull()                        # detect empty cells in the DF
dfI = df.interpolate(method='time')     # data interpolation (gap completion)
dfR = df.resample(rule = '30Min').mean() # change time step to 30 min with the mean methode
df.index = pd.to_datetime(TimeVector)    # use TimeVector to creat new index in the DF
df.plot()                          # use matplotlib to plot the DF (many options!)
```

### read/write data
```python
df.to_excel('filename1.xls')        # save DF in Excel file
df.to_csv('filename2.csv')          # save DF in text file
data1 = pd.read_csv('filename2.csv', sep=',', skiprows=0)   # read and load CSV (or .txt) file in a DF
data2 = pd.read_excel('filename1.xls') # read and load excel sheet in a DF
```
# Other interesting Package:
•	scipy -> optimization, linear algebra, statistics,…
•	scikit-learn, tensorflow -> machine learning
•	pylearn2, sklearn_theano -> neural network 
•	statsmodels -> statistic models (regressions, arx, …)
•	sympy -> formal calculus
•	deap, pyevolve -> genetic algorithm
•	time -> read computer time
•	multiprocessing -> calculs parallèles
•	requests -> internet access from python
•	csv, json -> read csv and json files (without pandas)
•	itertools -> combination, permutation
