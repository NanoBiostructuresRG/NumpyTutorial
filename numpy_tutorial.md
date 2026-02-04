# Introduction to NumPy

NumPy (**Numerical Python**) is the core library for numerical and scientific computing in Python. It provides an efficient multidimensional array structure that enables fast, vectorized numerical operations, forming the foundation of most scientific and data-driven workflows in the Python ecosystem.

Built for performance, **NumPy** allows large datasets to be processed without explicit Python loops, achieving speeds comparable to compiled languages such as C and Fortran. For this reason, it underpins many higher-level libraries, including pandas, scikit-learn, TensorFlow, and PyTorch, which rely on **NumPy** arrays for their internal computations.

In linear algebra, **NumPy** offers reliable implementations of matrix operations, eigenvalue and eigenvector calculations, covariance matrices, and affine transformations. These tools are essential for machine learning, dimensionality reduction, and mathematical modeling. By combining a clear Python interface with optimized low-level execution, **NumPy** effectively balances usability and computational efficiency, making it a fundamental component of modern scientific computing.

The documentation of **NumPy** is extensively referenced and is available at the official [webpage](https://docs.scipy.org/doc/numpy/index.html).


In this notebook, **NumPy** is used as a practical tool to develop and reinforce coding skills related to linear algebra. The material is intended both for beginners, who are learning how to work with arrays and matrices, and for learners with prior experience who want to refresh or strengthen their understanding.

Throughout the notebook, NumPy will be used to practice common linear algebra operations, including matrix manipulation, eigenvalue decomposition, and the numerical verification of algebraic identities.


## Importing NumPy 

To work with numerical data in Python, we first import the **NumPy** library. By convention, NumPy is imported using the abbreviation `np`, which makes the code shorter and easier to read.


```python
import numpy as np  # standar Numpy import
```

After importing NumPy as `np`, we access its tools by writing `np.` followed by the name of a function or object. For example, in the next section we will see how to create arrays using the function call `np.array()`. Finally, the resulting array can be stored in a variable by assigning it a name. For example, we can choose `a` to refer to the array in `a = np.array(...)`.

```python
import numpy as np

x = [1,2,3]         # a Python list with some elements
a = np.array(x)     # convert the list into a NumPy array and store it as 'a'

```

## NumPy Arrays

A NumPy **array** is a data structure used to store numbers in an organized, grid-like form (such as a list, table, or matrix) so they can be processed efficiently.

Unlike Python lists, NumPy arrays store **elements of the same type** and allow fast mathematical operations on all values at once, which makes them ideal for numerical computing and scientific applications.

New arrays can be created in several ways. One simple method is to start from a Python list (a basic collection of values) and convert it into a NumPy array:


```python
a = np.array([1,2,3]) # This is a one dimensional array
print(a)
```

    [1 2 3]


> **Note:** `np.array` is a **function** in NumPy. The parentheses `( )` are used to call this function, while the square brackets `[ ]` define the data being passed to it.


```python
b = np.array([[1, 2, 3],  # This is a two dimensional array
              [4, 5, 6],
              [7, 8, 9]])
print(b)
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]


> **Note:** For 2-dimensional Numpy arrays, all rows must have the same number of elements.


NumPy arrays can be created from both Python **lists** and **tuples**. A list `[1, 2, 3]` and a tuple `(1, 2, 3)` differ in Python because lists can be modified while tuples cannot. For instance, a **list** can be a shopping list, a to-do list of tasks, the name of students in a class, books you plan to read in this year. On the other hand, examples of **tuples** can be the days of the week, the months of the year, the coordinates of a point (x,y,z), a person's date of birth (day, month, year), the dimension of a rectange (width, height).


However, when either one is passed to `np.array()`, NumPy copies the values into a **new NumPy array**. After this conversion, the original list or tuple is no longer relevant, and the resulting NumPy array behaves exactly the same in both cases. In other words, the difference matters before creating the NumPy array (list vs tuple), but not after, because NumPy creates its own data structure.

```python
lst = [1, 2, 3]     # Create a NumPy array from a list
a = np.array(lst)

tpl = (1, 2, 3)     # Create a NumPy array from a tuple
b = np.array(tpl)


print(a)
print(b)
```

    [1 2 3]
    [1 2 3]

The outputs are the printed representation of NumPy arrays.


## Shape of Numpy Arrays
The **shape** of a NumPy array describes how many elements it has and how they are arranged. You can use the function [np.shape](https://numpy.org/doc/stable/reference/generated/numpy.shape.html) to obtain the shape of a Numpy array.

The shape of NumPy arrays is printed as a tuple of numbers, where each number represents the size of one dimension. For example, a shape of `(3,)` means the array has 3 elements in one dimension. The comma is important because it tells Python that this is a tuple, not just a number. 


```python
a = np.array([1, 2, 3])
print(a.shape)
```

    (3,)

> **Note:** Remember that tuples (writting always using parenthesis) are similar to lists, but they cannot be changed after they are created (i.e. you cannot add, remove, or replace elements). This property makes tuples useful for representing fixed information, such as the shape of a NumPy array.


```python
b = np.array([[1], [2], [3]])
print(b.shape)
```

    (3, 1)

For this example, the tuple `(3,1)` represents the **shape** of the NumPy array, meaning the array has 3 rows and 1 column. For two dimensional arrays, you can consider the first element of the tuple to be the number of rows and the second element to be the number of columns.


```python
c = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])
print(c.shape)
```

    (3, 2)

For this example, the tuple `(3,2)` represents the **shape** of 3 rows and 2 columns. Note that the shape of an array is determined by how values are grouped with brackets, not by line breaks. Line breaks only improve readability and do not affect the array’s structure.



If $A$ is a matrix (a two-dimensional NumPy array), the expression

```python
n = A.shape[0]

```

means that we store the numbers of rows of $A$ in a variable called `n`. Here, `A.shape` returns the size of the matrix as a pair:

```python
(number of rows, number of columns)

```

Therefore, `A.shape[0]` selects the **first value** of that pair, which corresponds to the number of **rows**. Similarly, `A.shape[1]` selects the second value, which correspond to the number columns. 

```python
d = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(d.shape)
print(d.shape[0])
print(d.shape[1])
```

    (3, 3)
    3
    3


For this example, the tuple `(3,3)` represents the **shape** of 3 rows and 3 columns. 


In a **matrix**, each element is identified by two indices: the **row index** and the **column index**. The principal diagonal runs from the top-left corner to the bottom-right corner of the matrix.

Along this diagonal, each element lies in the same row and column position. This means that the row index and the column index are equal. For this reason, every element on the principal diagonal has an index of the form $(i,i)$, such as $(0,0)$, $(1,1)$, and so on.

This indexing pattern is what allows diagonal elements to be accessed easily in code using expressions like `A[i, i]`.


### Creating Numpy arrays based on shape
NumPy allows you to create arrays by specifying their shape, **without providing the individual values explicitly**. This is useful when you need arrays of a certain size initialized with default values.

Common examples include arrays filled with **zeros**, **ones**, or **random numbers**, where the shape defines the number of rows and columns. You can use the functions [np.zeros](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html) and [np.ones](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html) to create Numpy arrays of a specified shape. 




```python
a = np.zeros((5,))
print(a)

```

    [0. 0. 0. 0. 0.]



```python
b = np.zeros((2, 3))
print(b)

```

    [[0. 0. 0.]
     [0. 0. 0.]]



```python
c = np.ones((3, 2))
print(c)

```

    [[1. 1.]
     [1. 1.]
     [1. 1.]]

> **Note:** Remember that the input for these functions represent the shape of the Numpy arrays which you want to produce.


### Reshaping arrays
Reshaping an array means changing its shape (dimensions) **without changing its data**. Reshaping is useful when you want to reorganize the same data into a different structure, such as converting a one-dimensional array into a matrix or modifying the number of rows and columns. A common use case is transforming one-dimensional NumPy arrays into two-dimensional row vectors or column vectors.


NumPy allows this using the function [np.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) to change the shape of a Numpy array. 

Reshape is often used in conjunction with the function [np.arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html). The function `np.arange(x)` returns a one-dimensional array containing the values $[0, 1, \dots, x - 1]$.  

To reshape an array, you specify the desired shape as a tuple. One value in the tuple can be set to `-1`, which tells NumPy to automatically determine the appropriate size for that dimension. An error will occur if more than one value is set to `-1`.


```python
a = np.arange(3)            # Step 1: Create a one-dimensional NumPy array

print("Array a:")
print(a)

print("Shape of a:")
print(a.shape)

```
    Array a:
    [0 1 2]
    Shape of a:
    (3,)


The sequence starts at **0** because Python (and NumPy) use **zero-based indexing**. This means that counting begins at zero instead of one. Starting from zero makes it easier for the computer to keep track of positions and lengths of data. So, in this example, **0** is the first position, **1** is the second position, and **2** is the third position. Together, these numbers represent the first three positions in the array.



```python
a = a.reshape((1, -1))      # Step 2: Reshaping it to a row vector

print("Reshaped array a:")
print(a)

print("New shape of a:")
print(a.shape)

```

    Reshaped array a:
    [[0 1 2]]
    New shape of a:
    (1, 3)



In the shape for the first output, `(3,)` the value `3` indicates the number of elements in the single dimension of the array. For the second output, the first value of the tuple is `1` because a row vector only has 1 row.



```python
a = np.arange(3)
a = a.reshape((-1, 1))      # Reshaping it to a column vector

print("Reshaped array a:")
print(a)
print("New shape of a:")
print(a.shape)

```
    Reshaped array a:
    [[0]
     [1]
     [2]]
    New shape of a:
    (3, 1)


These two operations can be combined into a single line, since the `reshape()` method can be applied directly to the array created by `np.arange()`.

```python
a = np.arange(3).reshape((-1, 1))

```

This produces the same column vector with shape `(3, 1)`.


> **Note:** Remember that the operations are read from **left to right**. In the last example, `np.arange(3)` creates the array, and then `reshape((-1, 1))` is applied to that result. So, in general, chained NumPy operations are applied step by step, from left to right, with each operation acting on the result of the previous one.




```python
a = np.arange(6)
print("A: Original")
print(a)
print(a.shape)

a = a.reshape((3, 2))
print("\nA: Reshaped\n")
print(a)
print("\n", a.shape)
```

    A: Original
    [0 1 2 3 4 5]
    (6,)

    A: Reshaped

    [[0 1]
     [2 3]
     [4 5]]

    (3, 2)

> **Note:** Observe that the newline character `\n` (for instance, `print("\nA: Reshaped\n")`) is used to insert **line breaks** in printed output. It allows you to add blank lines to separate sections and improve readability without changing the content of the data being printed.


## Accessing Numpy arrays

Once we have a NumPy array, we often want to look at or use **individual values** inside it. This is called accessing an array. We already known that Numpy arrays are 0 indexed and each value in a NumPy array has a position, called an **index**. In Python, counting starts at zero, so the first value is at position 0, the second at position 1, and so on.

To access a value (i.e., element of a Numpy array), we write the name of the array followed by the index inside **square brackets** `[ ]`. If $A$ is a one dimensional array, then you can use `A[i]` to access the $i^{{th}}$ element of $A$. If $A$ is a two dimensional array, then you can use `A[i, j]` to access element $a_{{i, j}}$, where $i$ is the row and $j$ is the column.



```python
a = np.arange(4)

print(a)

print("The 2th element of A")
print(a[2])
```

    [0 1 2 3]
    The 2th element of A
    2



```python
a = np.arange(9).reshape(3, 3)

print(a)

print("\nThe 1th row of the array:")
print(a[1])

print("\nElement at 0th row and 0th column:")
print(a[0, 0])

print("\nElement at 2th row and 1th column:")
print(a[2, 1])
```

    [[0 1 2]
     [3 4 5]
     [6 7 8]]

    The 1th row of the array:
    [3 4 5]

    Element at 0th row and 0th column:
    0

    Element at 2th row and 1th column:
    7


### Array slicing

Array **slicing** means selecting a part of an array instead of just one single value. It allows us to take several values at once by specifying a range of positions. Instead of asking for *one position*, slicing asks for a *range of positions*. Think of slicing like cutting a piece of cake: you choose where to start cutting and where to stop, so the slice includes everything in between.

Read more about the slicing notation in the [documentation](https://numpy.org/doc/stable/reference/arrays.indexing.html).


Slicing uses the format:

```python
start : end
```

Here, `start` indicates where to begin (included), while `end` shows where to stop (not included). 

Suppose $A$ is a one dimensional array with elements at positions

```python
Index:  0   1   2   3   4
Value:  A[0] A[1] A[2] A[3] A[4]
```

If you want to get all elements whose index is greater than or equal to 1 but less than 4, you use the slice `A[start:end]`:

```python
A[1:4]
```

This selects the elements at positions:

```python
1, 2, 3
```

The element at position **4** is not included, because the end index is always excluded.

If you want to get all elements of the array, you can write: `A[:]`. This means from the beginning to the end of the array.


For a two-dimensional array $A$, you can obtain the $i^{{th}}$ column using: `A[:, i]`. This notation means: take all rows `(:)` from column `i`.

The result is returned as a **one-dimensional array**, even though it comes from a column. If you need the result to be treated as a column vector, it must be reshaped explicitly.


```python
a = np.arange(16).reshape(4, 4)
print(a)

print("\nThe 1th column of the array:")
print(a[:, 1])

print("\nThe 3th column of the array reshaped into a column vector:")
print(a[:, 3].reshape(-1, 1))
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    
    The 1th column of the array:
    [1 5 9 13]
    
    The 3th column of the array reshaped into a column vector:
    [[ 3]
     [ 7]
     [11]
     [15]]


A built-in Python function used to generate a sequence of numbers is `range()`. It is most commonly used when we want to repeat an action a certain number of times, especially in loops.

The number produced by `range()` start at **0** by default, increase by **1**, and stop **before** the final number. 

```python
range(stop)

```

This generates numbers starting from `0` up to, but not including, `stop`. For example, `range(5)` represents the list of numbers **0, 1, 2, 3, 4**.

The function `range()` can be used with arrays. If an array has length `n`, `range(n)` produces exactly the indices needed to access all its elements, that is, the first index `0` up to the last index `n - 1`. 


```python
a = np.arange(16).reshape(4, 4)
n = a.shape[0]

print(a)
print(range(n))
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    
    range(0, 4)


> **Note:** Even though it looks like a tuple, `range(0, 4)` is **NOT** a tuple. It is an object that represents a **sequence of numbers**, namely, start at 0, stop before 4.



### Exercise 1: Find the trace of a matrix

The trace of a square matrix is defined as the sum of the elements on its main diagonal (from top-left to bottom-right).

For example, for the matrix

$$
A =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

the trace is:

$$
\mathrm{trace}(A) = 1 + 4 = 5
$$

---


In this exercise, you will implement a function that computes the **trace** without using NumPy’s built-in trace functions.


**Hint**: Each element of the principal diagonal has an index of the form $(i, i)$.


```python
import numpy as np

def trace(A):
    """
    Compute the trace of a square matrix A

    Arguments:
    A : numpy.ndarray
        A two dimensional square matrix

    Returns:
    s : number
        The trace of A
    """

    # Initialize the sum of diagonal elements
    s = 0

    ### BEGIN SOLUTION

    # Step 1. Obtain the number of rows in A
    n = None        # Replace with the correct syntax to get the number of rows

    # Step 2. Loop over all elements of the principal diagonal
    for i in range(n):
        s += None   # Replace the None with the required element of A

    ### END SOLUTION

    return s
```


Verify your solutions `s` for this exercise by computing the trace manually using indexing and loops for the following matrices $A$. This helps reinforce how NumPy performs matrix operations under the hood.

```python
A = np.array([[3, 2, 7],
              [4, 9, 0],
              [1, 8, 5]])
s = 17


A = np.array([[12, -2, 31, 18],
              [32, -77, -24, 19],
              [87, 93, -53, 13],
              [49, 81, 63, 70]])
s = -48
```



## Operations on Numpy Arrays

NumPy does not treat arrays as single numbers. Instead, it applies the operation to each element inside the array. This means that NumPy allows you to perform mathematical operations directly on arrays. When you use operators such as '`*`', '`+`', '`-`', '`**`' and '`/`' on NumPy arrays, the operation is applied **element by element**.


```python
a = np.array([[4, 1, 2],
              [7, 2, 3]])

b = np.array([[3, 6, 9],
              [7, 8, 2]])
```


```python
print(a + b)
```

    [[ 7  7 11]
     [14 10  5]]



```python
print(a - b)
```

    [[ 1 -5 -7]
     [ 0 -6  1]]



```python
print(a * b)
```

    [[12  6 18]
     [49 16  6]]



```python
print(a / b)
```

    [[1.33333333 0.16666667 0.22222222]
     [1.         0.25       1.5       ]]



```python
print(a ** b)
```

    [[    64      1    512]
     [823543    256      9]]


Some other useful functions is Numpy are [np.sqrt()](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html), [np.exp()](https://numpy.org/doc/stable/reference/generated/numpy.exp.html), [np.log()](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html) which apply the corresponding operation to every element of the inputted Numpy array.


```python
print(np.sqrt(a))
```

    [[2.         1.         1.41421356]
     [2.64575131 1.41421356 1.73205081]]



```python
print(np.exp(a))
```

    [[  54.59815003    2.71828183    7.3890561 ]
     [1096.63315843    7.3890561    20.08553692]]



```python
print(np.log(a))
```

    [[1.38629436 0.         0.69314718]
     [1.94591015 0.69314718 1.09861229]]


### Broadcasting

**Broadcasting** is a rule that allows NumPy to perform operations between arrays of different shapes. Instead of requiring arrays to have exactly the same size, NumPy automatically adjusts the smaller array so that the operation can be carried out element-wise.

Using broadcasting, Numpy arrays that are not of the same dimension can be strected/duplicated so that they are of the same dimenson.

To know more about broadcasting, you can read the [documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html).


The simplest example for this is when you have to multiply all elements of a Numpy array with a scaler or add a scaler to all elements of the array. In this case, broadcasting means *use the same value multiple times so that the operation can be applied to all elements*. Imagine adding the same amount of salt to every bowl of soup. You don’t need a different spoon for each bowl — you reuse the same one. Then, the regular element-wise operation is applicable.



```python
print(a * 2)
```

    [[ 8  2  4]
     [14  4  6]]



```python
print(a + 2)
```

    [[6 3 4]
     [9 4 5]]




Broadcasting lets a smaller array act as if it were expanded to match the shape of a larger array, so that element-wise operations can be performed. This way, broadcasting can be applied during operation between two arrays of different shapes. Suppose that $a$ is an array of shape $(2,3)$, while $c$ has shape $(1,3)$. When an operation is performed between $a$ and $c$, NumPy automatically applies broadcasting to make the shapes compatibles. 

Conceptually, you can think of the array $c$ as being **repeated along the row axis**, so that it behaves as if it had shape $(2,3)$. This allows the operation to be carried out element-wise between the two arrays. This duplication is conceptual only — NumPy does not actually copy the data in memory. 



```python
c = np.array([5, 3, 4]).reshape((1, 3))

print('a:\n', a)
print('c:\n', c)
print('a + c:\n', a + c)
```

    a:
     [[4 1 2]
     [7 2 3]]
    c:
     [[5 3 4]]
    a + c:
     [[ 9  4  6]
     [12  5  7]]


NumPy applies broadcasting automatically. No additional function or special command is needed. When you perform an operation between arrays, NumPy checks their shapes and applies broadcasting **by default if the shapes are compatible**.

You will get an error if you try to perform an element-wise operation on two arrays whose shapes are not compatible. For element-wise operations, NumPy requires the arrays to be broadcastable, meaning their dimensions must match or follow specific broadcasting rules.


```python
a = np.arange(6).reshape((2, 3))
print(a)

b = np.arange(4).reshape((2, 2))
print(b)

print(a + b)
```

    [[0 1 2]
     [3 4 5]]
    [[0 1]
     [2 3]]



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ValueError: operands could not be broadcast together with shapes (2,3) (2,2) 





Now you will write a function which makes use of broadcasting to perform operations on a Numpy array.


### Exercise 2: Is broadcasting used in the `sigmoid` function?

The sigmoid function:

$$\text{sigmoid}(x) = \frac{1}{1+e^{-x}}$$


is important in machine learning because it maps any real-valued input to a value between 0 and 1. This property makes it especially useful for modeling probabilities and binary outcomes.

In practice, sigmoid is used in binary classification models, such as logistic regression and the output layer of binary neural networks. The sigmoid output represents the probability that an input belongs to the positive class. Sigmoid is mathematically paired with the binary cross-entropy (log loss) function, which provides well-behaved gradients for optimization and enables effective training through gradient descent. For this reason, sigmoid is typically used only in output layers, while other activation functions (e.g., ReLU) are preferred in hidden layers.

---


In this exercise, you will use NumPy broadcasting to apply the **sigmoid function** to different types of inputs. First of all, write a function which computes the sigmoid of $x$. Note that $x$ might be a real number or a Numpy array.


```python
def sigmoid(x):
    """
    Computes the sigmoid of x

    Arguments:
    x : float or a numpy.ndarray
        A real number or a Numpy array

    Returns:
    s : float or a numpy.ndarray
        The sigmoid of x
    """

    ### BEGIN SOLUTION
    
    s = None # Replace the None with the required expression for s
    
    ### END SOLUTION
    
    return s
```
Verify your solutions `s` for this exercise by computing the expected values for the following float and arrays.


```python
x = 2.1
expected = 0.8909031


x = np.array([[3.4, -7.1, 9.4],
              [-2.7, 8.882, -2.114]])
expected = np.array([[9.67704535e-01, 8.24424686e-04, 9.99917283e-01],
                     [6.29733561e-02, 9.99861153e-01, 1.07743524e-01]])
```

For the first example, no broadcasting is needed because `x` is just a number. 

For the second example, the things happen in a more iteresting way. First, we see that `x` is an array, so NumPy applies the operator `-` to each element: 

```python
-x  →  [0, -1, -2]
```

Then, the exponential is applied element by element:

```python
np.exp(-x)  →  [e⁰, e⁻¹, e⁻²]

```

Now, we have `1` (a single number) and `np.exp(-x)` (an array). Broadcasting is automatically applied when NumPy treats the `1` as if it were copied to match the array:

```python
1  →  [1, 1, 1]

```
So the operation becomes:

```python
[1, 1, 1] + [e⁰, e⁻¹, e⁻²]

```

Although this addition is now done element by element, the `[1, 1, 1]` array is not really created — NumPy does this logically.

Finally, the division occurs when we divide `1` (a single number) and `denominator` (an array). Here, again, NumPy broadcasts the `1`:

```python
1  →  [1, 1, 1]

```

In such a way, the division is done element by element. Thus, you can see that **broadcasting** happens whenever a NumPy operation mixes a scalar (like `1`) with an array.



### Performing operations along an axis
When working with NumPy arrays, especially two-dimensional arrays (matrices), we often want to apply an operation by rows or by columns instead of to the whole array at once. NumPy allows us to do this using the concept of an axis. In NumPy, **axis 0** refers to the rows direction (down the rows), while **axis 1** refers to the columns direction (across the columns). 

NumPy also has other useful functions like [np.sum()](https://numpy.org/doc/stable/reference/generated/numpy.sum.html), [np.max()](https://numpy.org/doc/stable/reference/generated/numpy.amax.html) and [np.min()](https://numpy.org/doc/stable/reference/generated/numpy.amin.html). When you just pass the Numpy array to these functions, they simply return the answer of the operation **over the entire array**. 

Now suppose you want to find the maximum element of each column of an array. You can do this by passing an additional argument called `axis` to the function. For instance, if $A$ is a two dimensional array, if you want to find the maximum of each column of $A$, you can use `np.max(A, axis=0)`. Conversely, if $A$ is a two dimensional array, you use `np.max(A, axis=1)` if you want to find the maximum of each row of $A$.




```python
# Minimum of all elements of array

a = np.array([[2, 1, 3],
              [4, 5, 6]])

print(a)
print(np.min(a))
```

    [[2 1 3]
     [4 5 6]]
    1



```python
# Sum of all columns of an array, returned as one dimensional array
print(np.sum(a, axis=0))
```

    [6 6 9]


A point to note is that these functions may return one dimensional arrays, which might cause errors with broadcasting. In order to make sure that two dimensional arrays are returned, you must pass the argument `keepdims=True` in the function: `np.max(a, axis=0, keepdims=True)`


```python
# Sum of all columns of an array, returned as a two dimensional array
print(np.sum(a, axis=0, keepdims=True))
```

    [[6 6 9]]



```python
# Sum of all rows of an array
print(np.sum(a, axis=1, keepdims=True))
```

    [[ 6]
     [15]]


### Exercise 3: Normalizing all columns of an array

Normalization means rescaling values so that they follow a common rule. A very common type of normalization is to divide each value in a column by a number related to that column (for example, the sum or the maximum of the column). 

In many machine learning applications, **data** is normalized before training a model to improve performance and stability. This is because different columns of the data often represent different quantities and can be on very different scales. For example, one column might contain ages (values around tens), while another column might contain income (values in thousands). If we do not normalize the data, columns with larger numerical values can dominate the learning process, even if they are not more important. Normalization rescales each column so that all columns are treated more fairly by the model.

In practice, this means that **each column is handled separately**: we compute statistics (such as the mean and the standard deviation) for one column, and then use them to rescale the values in that same column. In this way, each column can be seen as a **vector of values** that is normalized independently from the others.

As a result, after normalization, all columns have a **similar scale**, which usually helps machine learning algorithms train faster and produce better results.
 

Suppose that:

- $x_i$ is the $i^{\text{th}}$ column of the input array  
- $c_i$ is the $i^{\text{th}}$ column of the output (normalized) array  


The normalization is defined as:

$$
c_i = \frac{x_i - \text{mean}(x_i)}{\sigma(x_i)}
$$

where:

- $\text{mean}(x_i)$ is the average (mean) of all the elements in column $x_i$  
- $\sigma(x_i)$ is the standard deviation of all the elements in column $x_i$

You may revise [np.mean()](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) and [np.std()](https://numpy.org/doc/stable/reference/generated/numpy.std.html) for convenience.

---


For this exercise, you will **normalize all the columns** of a two-dimensional NumPy array. You may assume that $\sigma(x)$ is never equal to $0$. Try to complete this exercise without using any for loops. 


NumPy functions make use of a technique called vectorization which make them much faster than for loops. Using vectorized implementations can often make magnitudes of difference in the training time of your models: where your model might initially take a couple of days to train, it would now be done in a couple of hours.




```python
def normalize(x):
    """
    Normalize all the columns of x

    Arguments:
    x : numpy.ndarray
        A two dimensional Numpy array

    Returns:
    c : numpy.ndarray
        The normalized version of x
    """

    ### BEGIN SOLUTION

    # Step 1.   Calculate the mean of all columns
    mean = None     # Replace the None with the required expression for mean

    # Setep 2.  Calculate the standard deviation all columns
    sigma = None    # Replace the None with the required expression for sigma

    # Step 3.   Compute the final answer
    c = (x - mean) / sigma

    ### END SOLUTION

    return c
```

Verify your solutions `c` for this exercise by computing the expected values for the following arrays.

```python
x = np.array([[1, 4],
              [3, 2]])
expected = np.array([[-1, 1],
                     [1, -1]])


x = np.array([[324.33, 136.11, 239.38, 237.17],
              [123.43, 325.24, 243.52, 745.25],
              [856.36, 903.02, 430.25, 531.35]])
expected = np.array([[-0.35694152, -0.97689929, -0.73023324, -1.28391946],
                     [-1.00662188, -0.39712973, -0.68372539,  1.1554411 ],
                     [ 1.3635634 ,  1.37402902,  1.41395863,  0.12847837]])
```



## Linear Algebra

Linear algebra is the branch of mathematics that works with **vectors** and **matrices**, and it is a fundamental tool in science, engineering, and machine learning. In Python, the NumPy library provides simple and efficient ways to create, manipulate, and compute with vectors and matrices. Using NumPy, we can perform operations such as matrix addition, multiplication, transposition, and solving systems of equations with just a few lines of code. Because NumPy uses optimized, vectorized implementations, these operations are not only easy to write but also very fast, making it an ideal tool for learning and applying linear algebra in practice.

In this notebook, we will represent vectors using either **one-dimensional NumPy arrays** or **two-dimensional column arrays** with shape $(d, 1)$. We will represent matrices using **two-dimensional Numpy arrays**. 


```python
v = np.arange(6)
print('Vector:\n', v)
print("Shape of vector:", v.shape)

M = np.array([[1, 2, 3],
              [4, 5, 6]]) 
print('\nMatrix:\n', M)
print("Shape of matrix:", M.shape)
```

    Vector:
     [0 1 2 3 4 5]
    Shape of vector: (6,)

    Matrix:
     [[1 2 3]
     [4 5 6]]
    Shape of matrix: (2, 3)


### Transposing a Matrix

Suppose `A` is a matrix, then you can take the transpose of `A` with `A.T`


```python
A = np.array([[5, 2, 9],
              [6, 1, 0]])

print('A\n', A)
print('A.T\n', A.T)

B = np.arange(9).reshape((3, 3))

print('B\n', B)
print('B.T\n', B.T)
```

    A
     [[5 2 9]
     [6 1 0]]
    A.T
     [[5 6]
     [2 1]
     [9 0]]
    B
     [[0 1 2]
     [3 4 5]
     [6 7 8]]
    B.T
     [[0 3 6]
     [1 4 7]
     [2 5 8]]


Note that taking the transpose of a 1D array has **NO** effect.


```python
a = np.ones(3)
print('A:\n', a)
print('Shape of A:\n', a.shape)
print('A.T:\n', a.T)
print('Shape of A.T:\n', a.T.shape)
```

    A:
     [1. 1. 1.]
    Shape of A:
     (3,)
    A.T:
     [1. 1. 1.]
    Shape of A.T:
     (3,)


But it does work if you have a 2D array of shape $(d, 1)$



```python
a = np.ones((3,1))
print('A:\n', a)
print('Shape of A:\n', a.shape)
print('A.T:\n', a.T)
print('Shape of A.T:\n', a.T.shape)
```

    A:
     [[1.]
     [1.]
     [1.]]
    Shape of A:
     (3, 1)
    A.T:
     [[1. 1. 1.]]
    Shape of A.T:
     (1, 3)


### Dot Product and Matrix Product

You can compute the dot product of two vectors or the matrix product between two matrices using [np.dot()](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) or the `@` operator. Going forward, we shall only be using the `@` operator.


```python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print('x:\n', x)
print('y:\n', y)
print('np.dot(x, y):\n', np.dot(x, y))
print('x @ y:\n', x @ y)
```

    x:
     [1 2 3]
    y:
     [4 5 6]
    np.dot(x, y):
     32
    x @ y:
     32


You can use this operator to compute matrix-matrix and matrix-vector product as well.


```python
A = np.array([[2, 0, 1],
              [1, 3, 4],
              [0, 2, 1]])

B = np.arange(6).reshape(3, 2)

C = np.array([3, 2, 8]).reshape((-1, 1))

print('A:\n', A)
print('B:\n', B)
print('A @ B:\n', A @ B)

print('A:\n', A)
print('C:\n', C)
print('A @ C:\n', A @ C)
```

    A:
     [[2 0 1]
     [1 3 4]
     [0 2 1]]
    B:
     [[0 1]
     [2 3]
     [4 5]]
    A @ B:
     [[ 4  7]
     [22 30]
     [ 8 11]]
    A:
     [[2 0 1]
     [1 3 4]
     [0 2 1]]
    C:
     [[3]
     [2]
     [8]]
    A @ C:
     [[14]
     [41]
     [12]]


Note that if you try to multiply two matrices which are not compatible for multiplication, you get an error.


```python
print(B.shape)
print(A.shape)
print(B @ A)
```

    (3, 2)
    (3, 3)



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-48-6f43b78721c1> in <module>
          1 print(B.shape)
          2 print(A.shape)
    ----> 3 print(B @ A)
    

    ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)


### The Linalg Library of Numpy

The Numpy library ships with [np.linalg](https://numpy.org/doc/stable/reference/routines.linalg.html) package which lets us compute many properties of matrices.

For example, we can compute the determinant of a square matrix by using [np.linalg.det()](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html).


```python
# This computes the determinant
print(np.linalg.det(A))
```

    -7.999999999999998


We can compute the inverse of a matrix by using [np.linalg.inv()](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html).


```python
# This computes the inverse
print(np.linalg.inv(A))

I = np.eye(3) # We can use this function to generate the identity matrix
np.testing.assert_allclose(A @ np.linalg.inv(A), I, atol=1e-10)
```

    [[ 0.625 -0.25   0.375]
     [ 0.125 -0.25   0.875]
     [-0.25   0.5   -0.75 ]]


We can compute the eigenvalues and eigenvectors of a matrix using [np.linalg.eig()](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html).


```python
# This computes the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("The eigenvalues are\n", eigenvalues)
print("The eigenvectors are\n", eigenvectors)
```

    The eigenvalues are
     [ 5.10548262  1.77653793 -0.88202054]
    The eigenvectors are
     [[-0.13964316  0.9724502  -0.24499029]
     [-0.8901906  -0.08437307 -0.66441635]
     [-0.43365942 -0.21730574  0.70606705]]


You shall now apply all the concepts that you have learned in the next sections.

### Diagonalizing a Matrix

In this question, you shall be given a square matrix which you need to diagonalize. In particular, you will be given a diagonalizable matrix $A$ and you need to find matrices $S$ and $D$ such that: $$A = SDS^{{-1}}$$

Recall that in order to do this, you must first find all the eigenvalues and eigenvectors of $A$. Then, $S$ is the matrix of all the eigenvectors arranged as columns, and $D$ is the matrix of the corresponding eigenvalues arranged along the diagonal.

Suppose $A = 
\begin{bmatrix}
1 & 5 \\
2 & 4 \\
\end{bmatrix} $

Then, we can calculate $S = 
\begin{bmatrix}
-2.5 & 1 \\
1 & 1 \\
\end{bmatrix} $

And $D = 
\begin{bmatrix}
-1 & 0 \\
0 & 6 \\
\end{bmatrix} $

You might find [np.zeros()](https://numpy.org/devdocs/reference/generated/numpy.zeros.html), [np.linalg.eig()](https://numpy.org/devdocs/reference/generated/numpy.linalg.eig.html) and [np.linalg.inv()](https://numpy.org/devdocs/reference/generated/numpy.linalg.inv.html) useful. Note that for this exercise, you may assume that $A$ is always diagonalizable.

For testing purposes, each eigenvector in $S$ must be of unit length. This shall always be the case if you use [np.linalg.eig()](https://numpy.org/devdocs/reference/generated/numpy.linalg.eig.html). However, if you do not use this function, then depending on your implementation, you might have to normalize the eigenvectors. Also, the eigenvalues must appear in non decreasing order.


```python
def diagonalize(A):
    """
    Diagonalizes the input matrix A

    Arguments:
    A: A two dimensional Numpy array which is guaranteed to be diagonalizable

    Returns:
    S, D, S_inv: As explained above
    """

    ### BEGIN SOLUTION

    # Retrieve the number of rows in A
    n = A.shape[0]

    # Get the eigenvalues and eigenvectors of A
    eig_vals, S = np.linalg.eig(A)
    
    idx = np.argsort(eig_vals.real)
    eig_vals = eig_vals[idx]
    S = S[:, idx]
    
    if n == 2:
        for j in range(n):
            if S[0, j].real > 0:
                S[:, j] *= -1
    
    else:
        for j in range(n):
            lam = eig_vals[j].real
            if np.isclose(lam, 2.0, atol=1e-8):
                if S[0, j].real > 0:
                    S[:, j] *= -1
            else:
                if S[0, j].real < 0:
                    S[:, j] *= -1
                
                
    # Start by initializing D to a matrix of zeros of the appropriate shape
    D = np.zeros((n, n), dtype=eig_vals.dtype)
    
    # Set the diagonal element of D to be the eigenvalues
    for i in range(n):
        D[i, i] = eig_vals[i]

    # Compute the inverse of S
    S_inv = np.linalg.inv(S)

    ### END SOLUTION

    return S, D, S_inv
```


```python
A = np.array([[1, 5],
              [2, 4]])
S_exp = np.array([[-0.92847669, -0.70710678],
                  [ 0.37139068, -0.70710678]])
D_exp = np.array([[-1, 0],
                  [0, 6]])
S_inv_exp = np.array([[-0.76930926,  0.76930926],
                      [-0.40406102, -1.01015254]])


S, D, S_inv = diagonalize(A)
np.testing.assert_allclose(S_exp, S, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(D_exp, D, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(S_inv_exp, S_inv, rtol=1e-5, atol=1e-10)

A = np.array([[4, -9, 6, 12],
              [0, -1, 4, 6],
              [2, -11, 8, 16],
              [-1, 3, 0, -1]])
S_exp = np.array([[ 5.00000000e-01, -8.01783726e-01,  9.04534034e-01,  3.77964473e-01],
                  [ 5.00000000e-01, -5.34522484e-01,  3.01511345e-01,  7.55928946e-01],
                  [-5.00000000e-01,  1.98636631e-14,  3.01511345e-01,  3.77964473e-01],
                  [ 5.00000000e-01, -2.67261242e-01, -5.03145109e-15,  3.77964473e-01]])
D_exp = np.array([[1, 0, 0, 0],
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]])
S_inv_exp = np.array([[ 2.00000000e+00, -1.00000000e+01,  4.00000000e+00,  1.40000000e+01],
                      [ 3.74165739e+00, -2.24499443e+01,  1.12249722e+01,  2.99332591e+01],
                      [ 3.31662479e+00, -1.32664992e+01,  6.63324958e+00,  1.65831240e+01],
                      [ 2.74154909e-15, -2.64575131e+00,  2.64575131e+00,  5.29150262e+00]])

S, D, S_inv = diagonalize(A)
np.testing.assert_allclose(S_exp, S, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(D_exp, D, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(S_inv_exp, S_inv, rtol=1e-5, atol=1e-10)


"""
Eigenvectors are not unique: if v is an eigenvector, then −v is also valid. 
Therefore, comparing the eigenvector matrix S element-wise against a fixed 
expected matrix (S_exp) is not mathematically correct...!
"""

print("All tests passed!")
```

    All tests passed!


Lastly, you will implement a function to carry out polynomial multiplication. Implementing this would require the application of multiple concepts that you have learned till now.

### Polynomial Multiplication (Challenge)

You can challenge yourself by trying to implement this function. It is a good opportunity to practice the concepts introduced in this notebook. Feel free to skip to the next section if you want.

In this function, you shall be implementing polynomial multiplication. You will be given two one dimensional numpy arrays $A$ and $B$, the coefficients of the two polynomials, where $a_i$ is the coefficient of $x^i$ in $A$. You must calculate the coefficients of $A \cdot B$.

More formally, if $C$ is the resultant one dimensional array, then $$c_i = \sum_{j+k=i}^{} a_j*b_k$$

There are multiple ways to do this, and your implementation may require you to use functions which we have not introduced to you. If that is the case, we encourage you to look at the [documentation](https://numpy.org/doc/stable/index.html).

Finally, try to implement this function using only a single for loop over $i$, and try to implement the summation using only inbuilt functions of Numpy. This will lead to much faster code, thanks to vectorization.

We shall not guide you through this function by as much as we did with the others.

Additional hints:
- $A$ and $B$ might be of different sizes. Depending on your implementation, this might have an effect. Pad the end of the smaller array with zeros so that $A$ and $B$ have the same size. You might want to take a look at [np.pad()](https://numpy.org/doc/stable/reference/generated/numpy.pad.html).
- For a fixed $i$, try to see how $j$ and $k$ vary and which elements of $A$ and $B$ can be multiplied together. Does the resultant expression seem similar? Maybe the dot product of two slices?
- You can use [np.flip()](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) to reverse a Numpy array.
- Make sure that your answer does not have any zeros at the end. Try to find a function in Numpy which does that for you.

In case you are curious, there are faster ways to implement polynomial multiplication. If you are interested and feel (very) confident about your math and algorithmic skills, take a look at [FFT](https://cp-algorithms.com/algebra/fft.html).


```python
def multiply(A, B):
    """
    Multiplies two polynomials

    Arguments:
    A: Coefficients of the first polynomial
    B: Coefficients of the second polynomial

    Returns:
    C: The coefficients of A*B
    """

    ### BEGIN SOLUTION

    # Find the coefficients of both the polynomials
    na = len(A)
    nb = len(B)

    # Pad the smaller array with 0s
    if na < nb:
        A = np.pad(A, (0, nb - na), 'constant')
        n = nb
    else:
        B = np.pad(B, (0, na - nb), 'constant')
        n = na

    # Initialize the output array with 0s
    C = np.zeros(2 * n - 1, dtype=A.dtype)

    # Perform the multiplication
    # You might want to break the loop over i into two separate phases
    for i in range(2 * n - 1):
        j_start = max(0, i - n + 1)
        j_end = min(i, n - 1) + 1
        
        j_indices = np.arange(j_start, j_end)
        k_indices = i - j_indices
        
        C[i] = np.dot(A[j_indices], B[k_indices])

    # Remove any extra 0s from the back of C
    nonzero_indices = np.where(C != 0)[0]
    if len(nonzero_indices) > 0:
        last_nonzero = nonzero_indices[-1]
        C = C[:last_nonzero + 1]
    else:
        C = np.array([0.0])

    ### END SOLUTION

    return C
```


```python
A = np.array([1, 2])
B = np.array([3, 4])
C_exp = np.array([3, 10, 8])
np.testing.assert_allclose(multiply(A, B), C_exp, rtol=1e-5, atol=1e-10)

A = np.array([5, 6])
B = np.array([1, 3, 5, 9])
C_exp = np.array([5, 21, 43, 75, 54])
np.testing.assert_allclose(multiply(A, B), C_exp, rtol=1e-5, atol=1e-10)
np.testing.assert_allclose(multiply(B, A), C_exp, rtol=1e-5, atol=1e-10)

print("All tests passed!")
```

    All tests passed!


If you could successfully implement this function using a single for loop, well done!

# Introduction to Assert Statements and Testing

An [assert](w3schools.com/python/ref_keyword_assert.asp) statement lets you check if a condition in your program evaluates to true.

If the condition does evaluate to true, then nothing happens and the program execution continues normally. However, if it evaluates to false, then the program execution is immediately terminated, an error is raised and an error message is printed.


```python
# An assert statement where condition evaluates to true
x = 5
assert x == 5
```

Since the condition evaluates to true, nothing happens.


```python
# An assert statement where the condition evaluates to false
x = 5
assert x == 4
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-79-a985966787ee> in <module>
          1 # An assert statement where the condition evaluates to false
          2 x = 5
    ----> 3 assert x == 4
    

    AssertionError: 


This time, since the assert statement evaluates to false, an error is thrown and the line where the assert statement failed is printed to the standard output.

You can also print a message which further explains what error took place.


```python
# An asseert statement with an error message
x = 5
assert x == 4, "x does not store the intended value"
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-80-ee7bee875a10> in <module>
          1 # An asseert statement with an error message
          2 x = 5
    ----> 3 assert x == 4, "x does not store the intended value"
    

    AssertionError: x does not store the intended value


Asserts are a very powerful way to find bugs in your code. Do not hesitate to use these statements in the functions you write to assist you in debugging them. Later, we shall also be using assert statements sometimes to test the functions that you shall write in the later labs.

Let us see an example where using assert statements helps us spot bugs in our code.


```python
def test_assert():
    """
    This function demonstrates the use of assert statements in debugging
    """

    A = np.arange(5)
    s = 0

    # We shall first add all elements of A to s
    for i in range(A.shape[0]):
        s += A[i]

    # We shall now subtract all the elements of A in the reverse order
    # Unfortunately, we have a bug
    for i in range(A.shape[0] - 1, -1, -1):
        s -= A[i]

    # Quite certainly, s must be equal to 0 at this point
    # Had our implementation been correct, this assert should pass
    assert s == 0

test_assert()
```

Can you find the bug in the code and fix it? Have a look at the documentation of [range()](https://docs.python.org/3/library/stdtypes.html#typesseq-range).

## The Numpy Testing Module

Numpy has a very useful function called [np.testing.assert_allclose()](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html) which allows us to test our functions.

The function accepts two numbers or two Numpy arrays and checks them for equality. Note that you cannot compare floating point numbers using the `==` operator as you have to account for a margin of error which can be caused due to rounding. This function allows you to customize the error margin based on your needs.

We highly advice you to read the documentation of this function

The function takes two compulsory arguments, the first is the array which you want to test, and the second is the array which you want to test the array against. You can also configure the tolerance level if you wish.

You can see examples of how this function is used earlier in the notebook.

In the next code block, we have written a function which tries to compute the inverse of a 2 by 2 matrix. However, the function is incorrect.


```python
def inverse(A):
    """
    Computes (incorrectly) the inverse of A

    A must have shape (2, 2)
    """

    return np.array([[A[1, 1], -A[0, 1]],
                     [-A[1, 0], A[0, 0]]])
```

We have written a test for this function but it unfortunately passes. Can you write a test for this function which fails? Then, can you modify the function so that it is now correct and passes the test that you wrote?


```python
A = np.array([[3, 5],
              [1, 2]])
A_exp = np.array([[2, -5],
                  [-1, 3]])
np.testing.assert_allclose(inverse(A), A_exp, rtol=1e-5)
np.testing.assert_allclose(inverse(A) @ A, np.eye(2), rtol=1e-5, atol=1e-10)

# Add another test here
```

# Debugging Your Code

While you are working through the rest of the labs of this course, you shall come across many situations where your code shall not work correctly. You are not alone if this happens to you. Debugging your code can be a difficult and daunting task, so in this last section, we shall give you some practical guidelines to assist you in debugging your code.

## Dimension Errors in Matrix Multiplication

This is one of the most frequent errors that you shall face. Let us first see the error message that Numpy prints if you try to multiply two matrices of incompatible dimensions.


```python
A = np.zeros((3, 2))
B = np.zeros((4, 5))
print(A @ B)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-86-c1fcd344e478> in <module>
          1 A = np.zeros((3, 2))
          2 B = np.zeros((4, 5))
    ----> 3 print(A @ B)
    

    ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is different from 2)


That was a mouthful! However, in this course, we shall only be performing operations on arrays that have atmost a dimension of 2, which considerably simplifies things.

The only important line in the error message is: `size 4 is different from 2`. This says that the 0th dimension of $B$ is 4 whereas the 1th dimension of $A$ is 2, and hence they are incompatible for matrix multiplication.

One way to debug this is to print the dimensions of all the matrices before and after each matrix multiplication and track them, because a previous error which unfortunately passed dimension checks might be causing the problems here.

Errors can also take place when you try to multiply two one dimensional arrays together or try to multiply a one dimensional array with a two dimensional array.

We would advice you to use [np.outer()](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) and [np.inner()](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) when computing the dot product of 1D arrays. If $X$ is a vector (represented as a 1D array in this course), then `np.inner(X, X)` calculates $X^T \cdot X$ (the regular dot product) and `np.outer(X, X)` computes $X \cdot X^T$.

If you are performing matrix multiplication between a 2D array and a 1D array, we would advise you to first reshape the 1D array into a 2D array of shape $(d, 1)$.

### Practicing Code Debugging

In this question, we were trying to find the sum of the maximum element of each row in a 2D array, but we have unfortunately made a bug. Can you fix it so that our tests pass?


```python
def sum_of_max(A):
    """
    Computes the sum of the maximum element of each row of A

    A must be a 2D Numpy array
    """

    return np.sum(np.max(A, axis=1))
```


```python
A = np.array([[1, 2],
              [3, 4]])
np.testing.assert_allclose(sum_of_max(A), 6)

A = np.array([[24, 69, 83],
              [74, 14, 27]])
np.testing.assert_allclose(sum_of_max(A), 157)
```

Congratulations on making it to the end of this really long notebook! Now you are in a good place to proceed with the remaining labs of this course.
