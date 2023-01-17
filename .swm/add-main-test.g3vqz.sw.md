---
id: g3vqz
title: Add main test
file_version: 1.1.1
app_version: 1.0.14
---

asd
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/__init__.md
```markdown
1      `
2      # Creating a .md File for a Script
3      
4      This example will show how to create a .md file for a script.
5      
6      ## Prerequisites
7      
8      Before you begin, you will need a text editor (such as Notepad, TextEdit, or Sublime Text) and the script you want to document.
9      
10     ## Step 1: Create the File
11     
12     Create a new text file and save it with the `.md` file extension.
13     
14     ## Step 2: Write the Script Documentation
15     
16     Write a brief overview of what the script does and the purpose it serves. Add any special instructions or considerations that may be needed.
17     
18     ## Step 3: Add Code Snippets
19     
20     Add code snippets to illustrate how the script works. For example, if you had a script that output a message to the console, you could add the following snippet:
21     
22     ```javascript
23     console.log('Hello World!');
24     ```
25     
26     ## Step 4: Save the File
27     
28     Once you have finished writing the file, save it in the same location as the script.
29     
30     ## Conclusion
31     
32     Creating a .md file for a script is a great way to provide documentation and illustrate how the script works. Follow the steps outlined in this tutorial to quickly create a .md file for your script.
```

<br/>

asd
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/bomb_enemy.md
```markdown
1      
2      # Maximum Enemies Killed by a Bomb
3      
4      Given a 2D grid, each cell is either a wall 'W', an enemy 'E' or empty '0' (the number zero), return the maximum enemies you can kill using one bomb. The bomb kills all the enemies in the same row and column from the planted point until it hits the wall since the wall is too strong to be destroyed. Note that you can only put the bomb at an empty cell.
5      
6      Example:
7      For the given grid 
8      
9      ```
10     0 E 0 0
11     E 0 W E
12     0 E 0 0
13     ```
14     
15     return 3. (Placing a bomb at (1,1) kills 3 enemies)
16     
17     ## Solution
18     
19     ```python
20     def max_killed_enemies(grid):
21         if not grid:
22             return 0
23         m, n = len(grid), len(grid[0])
24         max_killed = 0
25         row_e, col_e = 0, [0] * n
26         # iterates over all cells in the grid
27         for i in range(m):
28             for j in range(n):
29                 # makes sure we are next to a wall.
30                 if j == 0 or grid[i][j-1] == 'W':
31                     row_e = row_kills(grid, i, j)
32                 # makes sure we are next to a wall.
33                 if i == 0 or grid[i-1][j] == 'W':
34                     col_e[j] = col_kills(grid, i, j)
35                 # makes sure the cell contains a 0
36                 if grid[i][j] == '0':
37                     # updates the variable
38                     max_killed = max(max_killed, row_e + col_e[j])
39     
40         return max_killed
41     
42     
43     # calculate killed enemies for row i from column j
44     def row_kills(grid, i, j):
45         num = 0
46         len_row = len(grid[0])
47         while j < len_row and grid[i][j] != 'W':
48             if grid[i][j] == 'E':
49                 num += 1
50             j += 1
51         return num
52     
53     
54     # calculate killed enemies for  column j from row i
55     def col_kills(grid, i, j):
56         num = 0
57         len_col = len(grid)
58         while i < len_col and grid[i][j] != 'W':
59             if grid[i][j] == 'E':
60                 num += 1
61             i += 1
62         return num
63     ```
64     
65     ## Test Cases
66     
67     ```python
68     import unittest
69     
70     
71     class TestBombEnemy(unittest.TestCase):
72         def test_3x4(self):
73             grid1 = [["0", "E", "0", "0"],
74                      ["E", "0", "W", "E"],
75                      ["0", "E", "0", "0"]]
76             self.assertEqual(3, max_killed_enemies(grid1))
77     
78         def test_4x4(self):
79             grid1 = [
80                     ["0", "E", "0", "E"],
81                     ["E", "E", "E", "0"],
82                     ["E", "0", "W", "E"],
83                     ["0", "E", "0", "0"]]
84             grid2 = [
85                     ["0", "0", "0", "E"],
86                     ["E", "0", "0", "0"],
87                     ["E", "0", "W", "E"],
88                     ["0", "E", "0", "0"]]
89             self.assertEqual(5, max_killed_enemies(grid1))
90             self.assertEqual(3, max_killed_enemies(grid2))
91     
92     
93     if __name__ == "__main__":
94         unittest.main()
95     ```
```

<br/>

asd
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/cholesky_matrix_decomposition.md
```markdown
1      
2      # Cholesky Matrix Decomposition
3      
4      Cholesky matrix decomposition is a technique used to find the decomposition of a Hermitian positive-definite matrix A into matrix V, so that V * V* = A, where V* denotes the conjugate transpose of L. The dimensions of the matrix A must match. This method is mainly used for numeric solution of linear equations Ax = b.
5      
6      ## Example
7      
8      Input matrix A:
9      
10     ```
11     [[  4,  12, -16],
12      [ 12,  37, -43],
13      [-16, -43,  98]]
14     ```
15     
16     Result:
17     
18     ```
19     [[2.0, 0.0, 0.0],
20     [6.0, 1.0, 0.0],
21     [-8.0, 5.0, 3.0]]
22     ```
23     
24     ## Time Complexity
25     
26     The time complexity of this algorithm is O(n^3), specifically about (n^3)/3.
27     
28     ## Algorithm
29     
30     ```python
31     def cholesky_decomposition(A):
32         """
33         :param A: Hermitian positive-definite matrix of type List[List[float]]
34         :return: matrix of type List[List[float]] if A can be decomposed,
35         otherwise None
36         """
37         n = len(A)
38         for ai in A:
39             if len(ai) != n:
40                 return None
41         V = [[0.0] * n for _ in range(n)]
42         for j in range(n):
43             sum_diagonal_element = 0
44             for k in range(j):
45                 sum_diagonal_element = sum_diagonal_element + math.pow(V[j][k], 2)
46             sum_diagonal_element = A[j][j] - sum_diagonal_element
47             if sum_diagonal_element <= 0:
48                 return None
49             V[j][j] = math.pow(sum_diagonal_element, 0.5)
50             for i in range(j+1, n):
51                 sum_other_element = 0
52                 for k in range(j):
53                     sum_other_element += V[i][k]*V[j][k]
54                 V[i][j] = (A[i][j] - sum_other_element)/V[j][j]
55         return V
56     ```
```

<br/>

asdddd
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/comd_transform.md
```markdown
1      
2      # Rotating Matrices
3      
4      Matrices can be rotated in four different ways: clockwise, counterclockwise, top-left invert, and bottom-left invert. This script will demonstrate how to rotate a matrix using Python. 
5      
6      We begin by defining four functions: `rotate_clockwise`, `rotate_counterclockwise`, `top_left_invert`, and `bottom_left_invert`. The functions take a matrix as an argument and return a new matrix that is rotated in the specified direction. 
7      
8      ## `rotate_clockwise`
9      
10     This function rotates a matrix clockwise by reversing the rows and then iterating over each column.
11     
12     ```python
13     def rotate_clockwise(matrix):
14         new = []
15         for row in reversed(matrix):
16             for i, elem in enumerate(row):
17                 try:
18                     new[i].append(elem)
19                 except IndexError:
20                     new.insert(i, [])
21                     new[i].append(elem)
22         return new
23     ```
24     
25     ## `rotate_counterclockwise`
26     
27     This function rotates a matrix counterclockwise by reversing the columns and then iterating over each row.
28     
29     ```python
30     def rotate_counterclockwise(matrix):
31         new = []
32         for row in matrix:
33             for i, elem in enumerate(reversed(row)):
34                 try:
35                     new[i].append(elem)
36                 except IndexError:
37                     new.insert(i, [])
38                     new[i].append(elem)
39         return new
40     ```
41     
42     ## `top_left_invert`
43     
44     This function inverts a matrix top-left by iterating over each column and then appending the elements to a new matrix.
45     
46     ```python
47     def top_left_invert(matrix):
48         new = []
49         for row in matrix:
50             for i, elem in enumerate(row):
51                 try:
52                     new[i].append(elem)
53                 except IndexError:
54                     new.insert(i, [])
55                     new[i].append(elem)
56         return new
57     ```
58     
59     ## `bottom_left_invert`
60     
61     This function inverts a matrix bottom-left by iterating over each column in reverse order and then appending the elements to a new matrix.
62     
63     ```python
64     def bottom_left_invert(matrix):
65         new = []
66         for row in reversed(matrix):
67             for i, elem in enumerate(reversed(row)):
68                 try:
69                     new[i].append(elem)
70                 except IndexError:
71                     new.insert(i, [])
72                     new[i].append(elem)
73         return new
74     ```
75     
76     We can now test the functions. We define a matrix and then print the initial matrix and the rotated matrices.
77     
78     ```python
79     if __name__ == '__main__':
80         def print_matrix(matrix, name):
81             print('{}:\n['.format(name))
82             for row in matrix:
83                 print('  {}'.format(row))
84             print(']\n')
85     
86         matrix = [
87             [1, 2, 3],
88             [4, 5, 6],
89             [7, 8, 9],
90         ]
91     
92         print_matrix(matrix, 'initial')
93         print_matrix(rotate_clockwise(matrix), 'clockwise')
94         print_matrix(rotate_counterclockwise(matrix), 'counterclockwise')
95         print_matrix(top_left_invert(matrix), 'top left invert')
96         print_matrix(bottom_left_invert(matrix), 'bottom left invert')
97     ```
98     
99     When we run the script, we get the following output:
100    
101    ```
102    initial:
103    [
104      [1, 2, 3]
105      [4, 5, 6]
106      [7, 8, 9]
107    ]
108    
109    clockwise:
110    [
111      [7, 4, 1]
112      [8, 5, 2]
113      [9, 6, 3]
114    ]
115    
116    counterclockwise:
117    [
118      [3, 6, 9]
119      [2, 5, 8]
120      [1, 4, 7]
121    ]
122    
123    top left invert:
124    [
125      [1, 4, 7]
126      [2, 5, 8]
127      [3, 6, 9]
128    ]
129    
130    bottom left invert:
131    [
132      [9, 8, 7]
133      [6, 5, 4]
134      [3, 2, 1]
135    ]
136    ```
137    
138    As you can see, the functions have successfully rotated the matrix in the desired directions.
```

<br/>

2222
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/count_paths.md
```markdown
1      
2      # Count the Number of Unique Paths from a[0][0] to a[m-1][n-1]
3      
4      Finding the number of unique paths from the starting point of a matrix to the end point can be a challenging problem. We are allowed to move either right or down from a cell in the matrix. There are two approaches to solve this problem:
5      
6      ## Recursion
7      
8      This approach recursively calls the same function starting from the end point of the matrix, a[m-1][n-1], and moving upwards and leftwards. At each recursive call, the path count of both recursions is added and then returned.
9      
10     Time Complexity: O(mn)  
11     Space Complexity: O(mn)
12     
13     ## Dynamic Programming
14     
15     This approach starts from the starting point of the matrix, a[0][0], and stores the count in a count matrix. The number of ways to reach a[i][j] is calculated by taking the sum of the number of ways to reach a[i-1][j] and a[i][j-1]. The final answer is returned from count[m-1][n-1].
16     
17     Time Complexity: O(mn)  
18     Space Complexity: O(mn)
19     
20     Below is the Python code for the Dynamic Programming approach:
21     
22     ```python
23     def count_paths(m, n):
24         if m < 1 or n < 1:
25             return -1
26         count = [[None for j in range(n)] for i in range(m)]
27     
28         # Taking care of the edge cases- matrix of size 1xn or mx1
29         for i in range(n):
30             count[0][i] = 1
31         for j in range(m):
32             count[j][0] = 1
33     
34         for i in range(1, m):
35             for j in range(1, n):
36                 # Number of ways to reach a[i][j] = number of ways to reach
37                 #                                   a[i-1][j] + a[i][j-1]
38                 count[i][j] = count[i - 1][j] + count[i][j - 1]
39     
40         print(count[m - 1][n - 1])
41     
42     
43     def main():
44         m, n = map(int, input('Enter two positive integers: ').split())
45         count_paths(m, n)
46     
47     
48     if __name__ == '__main__':
49         main()
50     ```
```

<br/>

d
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/crout_matrix_decomposition.md
<!-- collapsed -->

```markdown
1      
2      # Crout Matrix Decomposition
3      
4      Crout matrix decomposition is used to find two matrices that, when multiplied, give our input matrix, so L * U = A. L stands for lower and L has non-zero elements only on diagonal and below. U stands for upper and U has non-zero elements only on diagonal and above. This can for example be used to solve systems of linear equations. The last if is used to avoid dividing by zero.
5      
6      ## Example
7      We input the A matrix:
8      ```
9      [[1,2,3],
10     [3,4,5],
11     [6,7,8]]
12     ```
13     We get:
14     L = 
15     ```
16     [1.0,  0.0, 0.0]
17     [3.0, -2.0, 0.0]
18     [6.0, -5.0, 0.0]
19     ```
20     U = 
21     ```
22     [1.0,  2.0, 3.0]
23     [0.0,  1.0, 2.0]
24     [0.0,  0.0, 1.0]
25     ```
26     We can check that L * U = A.
27     
28     ## Complexity
29     I think the complexity should be O(n^3).
30     
31     ## Code
32     ```python
33     def crout_matrix_decomposition(A):
34         n = len(A)
35         L = [[0.0] * n for i in range(n)]
36         U = [[0.0] * n for i in range(n)]
37         for j in range(n):
38             U[j][j] = 1.0
39             for i in range(j, n):
40                 alpha = float(A[i][j])
41                 for k in range(j):
42                     alpha -= L[i][k]*U[k][j]
43                 L[i][j] = float(alpha)
44             for i in range(j+1, n):
45                 tempU = float(A[j][i])
46                 for k in range(j):
47                     tempU -= float(L[j][k]*U[k][i])
48                 if int(L[j][j]) == 0:
49                     L[j][j] = float(0.1**40)
50                 U[j][i] = float(tempU/L[j][j])
51         return (L, U)
52     ```
```

<br/>

333
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/matrix_exponentiation.md
<!-- collapsed -->

```markdown
1      
2      # Matrix Exponentiation
3      
4      Matrix exponentiation is a technique used to calculate a matrix raised to the power of n, where n is a positive integer. This technique can be used to solve problems in various fields, such as graph theory.
5      
6      ## Algorithm
7      
8      The algorithm for matrix exponentiation is based on repeated squaring. We start by raising the matrix to the power of 2, and then raising the result to the power of 4, then 8 and so on, until we reach the desired power.
9      
10     For example, to calculate the matrix raised to the power of 9, we can calculate it as follows:
11     
12     `matrix_exponentiation(mat, 9)`
13     
14     = `multiply(matrix_exponentiation(mat, 8), mat)`
15     
16     = `multiply(multiply(matrix_exponentiation(mat, 4), matrix_exponentiation(mat, 4)), mat)`
17     
18     = `multiply(multiply(multiply(matrix_exponentiation(mat, 2), matrix_exponentiation(mat, 2)), matrix_exponentiation(mat, 2)), mat)`
19     
20     ## Code Snippet
21     
22     The code snippet below implements the matrix exponentiation algorithm in Python.
23     
24     ```python
25     def multiply(matA: list, matB: list) -> list:
26         """
27         Multiplies two square matrices matA and matB od size n x n
28         Time Complexity: O(n^3)
29         """
30         n = len(matA)
31         matC = [[0 for i in range(n)] for j in range(n)]
32     
33         for i in range(n):
34             for j in range(n):
35                 for k in range(n):
36                     matC[i][j] += matA[i][k] * matB[k][j]
37     
38         return matC
39     
40     
41     def identity(n: int) -> list:
42         """
43         Returns the Identity matrix of size n x n
44         Time Complexity: O(n^2)
45         """
46         I = [[0 for i in range(n)] for j in range(n)]
47     
48         for i in range(n):
49             I[i][i] = 1
50     
51         return I
52     
53     
54     def matrix_exponentiation(mat: list, n: int) -> list:
55         """
56         Calculates mat^n by repeated squaring
57         Time Complexity: O(d^3 log(n))
58                          d: dimension of the square matrix mat
59                          n: power the matrix is raised to
60         """
61         if n == 0:
62             return identity(len(mat))
63         elif n % 2 == 1:
64             return multiply(matrix_exponentiation(mat, n - 1), mat)
65         else:
66             tmp = matrix_exponentiation(mat, n // 2)
67             return multiply(tmp, tmp)
68     ```
69     
70     ## Time Complexity
71     
72     The time complexity of the matrix exponentiation algorithm is O(d<sup>3</sup> log(n)), where d is the dimension of the square matrix mat and n is the power the matrix is raised to.
```

<br/>

33333
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/matrix_inversion.md
```markdown
1      
2      # Inverting an n x n Matrix
3      
4      Inverting an invertible n x n matrix is a four step process that can be done with the following steps:
5      
6      1. Calculate the matrix of minors: Create an n x n matrix by considering each position in the original matrix in turn. Exclude the current row and column and calculate the determinant of the remaining matrix, then place that value in the current position's equivalent in the matrix of minors.
7      2. Create the matrix of cofactors: Take the matrix of minors and multiply alternate values by -1 in a checkerboard pattern.
8      3. Adjugate: Hold the top left to bottom right diagonal constant, but swap all other values over it.
9      4. Multiply the adjugated matrix by 1 / the determinant of the original matrix.
10     
11     This code combines steps 1 and 2 into one method to reduce traversals of the matrix:
12     
13     ```python
14     def get_matrix_of_minors(m):
15         """get the matrix of minors and alternate signs"""
16         matrix_of_minors = [[0 for i in range(len(m))] for j in range(len(m))]
17         for row in range(len(m)):
18             for col in range(len(m[0])):
19                 if (row + col) % 2 == 0:
20                     sign = 1
21                 else:
22                     sign = -1
23                 matrix_of_minors[row][col] = sign * get_determinant(get_minor(m, row, col))
24         return matrix_of_minors
25     ```
26     
27     For a 2 x 2 matrix, inversion is simpler. The code for this is as follows:
28     
29     ```python
30     if len(m) == 2:
31         # simple case
32         multiplier = 1 / get_determinant(m)
33         inverted = [[multiplier] * len(m) for n in range(len(m))]
34         inverted[0][1] = inverted[0][1] * -1 * m[0][1]
35         inverted[1][0] = inverted[1][0] * -1 * m[1][0]
36         inverted[0][0] = multiplier * m[1][1]
37         inverted[1][1] = multiplier * m[0][0]
38         return inverted
39     ```
40     
41     Possible edge cases: will not work for 0x0 or 1x1 matrix, though these are trivial to calculate without use of this file.
```

<br/>

sssss
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/multiply.md
```markdown
1      
2      # Matrix Multiplication Algorithm
3      This algorithm takes two compatible two dimensional matrices and returns their product. The space complexity of this algorithm is **O(n<sup>2</sup>)**. 
4      
5      The possible edge case for this algorithm is when the number of columns of the multiplicand is not consistent with the number of rows of the multiplier. This will raise an exception.
6      
7      ## Code Snippet
8      ```python
9      def multiply(multiplicand: list, multiplier: list) -> list:
10         """
11         :type A: List[List[int]]
12         :type B: List[List[int]]
13         :rtype: List[List[int]]
14         """
15         multiplicand_row, multiplicand_col = len(
16             multiplicand), len(multiplicand[0])
17         multiplier_row, multiplier_col = len(multiplier), len(multiplier[0])
18         if(multiplicand_col != multiplier_row):
19             raise Exception(
20                 "Multiplicand matrix not compatible with Multiplier matrix.")
21         # create a result matrix
22         result = [[0] * multiplier_col for i in range(multiplicand_row)]
23         for i in range(multiplicand_row):
24             for j in range(multiplier_col):
25                 for k in range(len(multiplier)):
26                     result[i][j] += multiplicand[i][k] * multiplier[k][j]
27         return result
28     ```
```

<br/>

asd
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/rotate_image.md
```markdown
1      
2      # Rotating an Image by 90 Degrees
3      
4      Rotating an image by 90 degrees is a common task in computer vision and image processing. In this article, we will discuss how to rotate an image by 90 degrees (clockwise) using the Python language.
5      
6      ## Problem Description
7      
8      Given an n x n 2D matrix representing an image, we wish to rotate the image by 90 degrees (clockwise).
9      
10     ## Solution
11     
12     We can solve this problem in-place by first reversing the matrix up to down, then swapping the symmetry of the elements.
13     
14     ### Algorithm
15     
16     1. Reverse the matrix up to down
17     2. Swap the symmetry of the elements
18     
19     ### Code Snippet
20     
21     ```python
22     def rotate(mat):
23         if not mat:
24             return mat
25         mat.reverse()
26         for i in range(len(mat)):
27             for j in range(i):
28                 mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
29         return mat
30     ```
31     
32     ### Example
33     
34     Let's consider the following example:
35     
36     ```
37     1 2 3
38     4 5 6
39     7 8 9
40     ```
41     
42     Applying the above algorithm, we get the following result:
43     
44     ```
45     7 4 1
46     8 5 2
47     9 6 3
48     ```
49     
50     ## Conclusion
51     
52     In this article, we discussed how to rotate an image by 90 degrees (clockwise) using the Python language. We used an algorithm that reversed the matrix up to down, then swapped the symmetry of the elements.
```

<br/>

asddddd
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/search_in_sorted_matrix.md
```markdown
1      
2      # Searching in a Sorted Matrix
3      Given a (row-wise and column-wise) sorted matrix, we need to search a key in it.
4      
5      The time complexity of searching in a sorted matrix is **O(m + n)**, where m is the number of rows and n is the number of columns in the matrix. 
6      
7      ## Algorithm
8      
9      1. Start with the top right element in the matrix.
10     2. If the key is larger than the current element, move one column to the left.
11     3. If the key is smaller than the current element, move one row down.
12     4. Repeat steps 2 and 3 until the key is found or the matrix boundaries are reached.
13     
14     ## Implementation
15     
16     ```
17     def search_in_a_sorted_matrix(mat, m, n, key):
18         i, j = m-1, 0
19         while i >= 0 and j < n:
20             if key == mat[i][j]:
21                 print('Key %s found at row- %s column- %s' % (key, i+1, j+1))
22                 return
23             if key < mat[i][j]:
24                 i -= 1
25             else:
26                 j += 1
27         print('Key %s not found' % (key))
28     ```
29     
30     Let us consider a matrix `mat` as shown below:
31     
32     ```
33     mat = [
34                [2, 5, 7],
35                [4, 8, 13],
36                [9, 11, 15],
37                [12, 17, 20]
38               ]
39     ```
40     
41     To search for the key `13` in the matrix, we have to execute the following code:
42     
43     ```
44     key = 13
45     search_in_a_sorted_matrix(mat, len(mat), len(mat[0]), key)
46     ```
47     
48     The output of the above code would be:
49     
50     ```
51     Key 13 found at row- 2 column- 3
52     ```
```

<br/>

ddddddd
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/sort_matrix_diagonally.md
```markdown
1      
2      # Sort Diagonally
3      
4      Given a m * n matrix `mat` of integers, you are tasked with sorting it diagonally in ascending order from the top-left to the bottom-right and then returning the sorted array. 
5      
6      For example, given the following matrix:
7      
8      ```
9      mat = [
10         [3,3,1,1],
11         [2,2,1,2],
12         [1,1,1,2]
13     ]
14     ```
15     
16     The expected output is:
17     
18     ```
19     [
20         [1,1,1,1],
21         [1,2,2,2],
22         [1,2,3,3]
23     ]
24     ```
25     
26     ## Solution
27     
28     We can use a heap to sort each diagonal of the matrix. To do this, we can iterate through each row and column and add each element to the heap. Then, we can sort the heap and update each element in the matrix. 
29     
30     We can represent this algorithm in Python as follows:
31     
32     ```python
33     from heapq import heappush, heappop
34     from typing import List
35     
36     def sort_diagonally(mat: List[List[int]]) -> List[List[int]]:
37         # If the input is a vector, return the vector
38         if len(mat) == 1 or len(mat[0]) == 1:
39             return mat
40     
41         # Rows + columns - 1
42         # The -1 helps you to not repeat a column
43         for i in range(len(mat)+len(mat[0])-1):
44             # Process the rows
45             if i+1 < len(mat):
46                 # Initialize heap, set row and column
47                 h = []
48                 row = len(mat)-(i+1)
49                 col = 0
50     
51                 # Traverse diagonally, and add the values to the heap
52                 while row < len(mat):
53                     heappush(h, (mat[row][col]))
54                     row += 1
55                     col += 1
56     
57                 # Sort the diagonal
58                 row = len(mat)-(i+1)
59                 col = 0
60                 while h:
61                     ele = heappop(h)
62                     mat[row][col] = ele
63                     row += 1
64                     col += 1
65             else:
66                 # Process the columns
67                 # Initialize heap, row and column
68                 h = []
69                 row = 0
70                 col = i - (len(mat)-1)
71     
72                 # Traverse Diagonally
73                 while col < len(mat[0]) and row < len(mat):
74                     heappush(h, (mat[row][col]))
75                     row += 1
76                     col += 1
77     
78                 # Sort the diagonal
79                 row = 0
80                 col = i - (len(mat)-1)
81                 while h:
82                     ele = heappop(h)
83                     mat[row][col] = ele
84                     row += 1
85                     col += 1
86     
87         # Return the updated matrix
88         return mat
89     ```
90     
91     We start by checking if the input is a vector. If it is, we simply return the vector. Otherwise, we iterate through each row and column, adding each element to the heap. We then sort the heap and update each element in the matrix. 
92     
93     Finally, we return the updated matrix.
94     
95     ## Complexity
96     
97     The time complexity of this algorithm is O(m<sup>2</sup>n<sup>2</sup>) as we need to iterate through each row and column and add each element to the heap. The space complexity is O(m+n) as we need to store the elements in the heap.
```

<br/>

123
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/sparse_dot_vector.md
```markdown
1      
2      
3      # Storing Sparse Vectors and Calculating Dot Product
4      
5      When dealing with large sparse vectors, which contain a lot of zeros and doubles, it can be helpful to use a data structure to store them. In this example, we will go over a method that can be used to store such vectors and calculate their dot product. 
6      
7      ## Vector to Index/Value List
8      
9      The first step is to create a list which contains the index and values of the vector. To do this, we can use a list comprehension to loop through the vector and store the index and the value for each non-zero value in the vector:
10     
11     ```python
12     def vector_to_index_value_list(vector):
13         return [(i, v) for i, v in enumerate(vector) if v != 0.0]
14     ```
15     
16     ## Calculating the Dot Product
17     
18     Once the vector has been converted to an index/value list, we can calculate the dot product between two vectors. To do this, we use a while loop to traverse through both index/value lists and add the product of their corresponding values to the product variable. Once we reach the end of one of the lists, we can return the product. 
19     
20     ```python
21     def dot_product(iv_list1, iv_list2):
22     
23         product = 0
24         p1 = len(iv_list1) - 1
25         p2 = len(iv_list2) - 1
26     
27         while p1 >= 0 and p2 >= 0:
28             i1, v1 = iv_list1[p1]
29             i2, v2 = iv_list2[p2]
30     
31             if i1 < i2:
32                 p1 -= 1
33             elif i2 < i1:
34                 p2 -= 1
35             else:
36                 product += v1 * v2
37                 p1 -= 1
38                 p2 -= 1
39     
40         return product
41     ```
42     
43     ## Testing the Code
44     
45     To test this code, we can create a test to check that the dot product of two simple vectors is correct, and then time how long it takes to calculate the dot product of two large sparse vectors. 
46     
47     For the first test, we can create two simple vectors and calculate their dot product:
48     
49     ```python
50     def __test_simple():
51         print(dot_product(vector_to_index_value_list([1., 2., 3.]),
52                           vector_to_index_value_list([0., 2., 2.])))
53         # 10
54     ```
55     
56     For the second test, we can create two large sparse vectors and time how long it takes for the dot product to be calculated:
57     
58     ```python
59     def __test_time():
60         vector_length = 1024
61         vector_count = 1024
62         nozero_counut = 10
63     
64         def random_vector():
65             import random
66             vector = [0 for _ in range(vector_length)]
67             for i in random.sample(range(vector_length), nozero_counut):
68                 vector[i] = random.random()
69             return vector
70     
71         vectors = [random_vector() for _ in range(vector_count)]
72         iv_lists = [vector_to_index_value_list(vector) for vector in vectors]
73     
74         import time
75     
76         time_start = time.time()
77         for i in range(vector_count):
78             for j in range(i):
79                 dot_product(iv_lists[i], iv_lists[j])
80         time_end = time.time()
81     
82         print(time_end - time_start, 'seconds')
83     ```
84     
85     ## Conclusion
86     
87     In this example, we went over a method for storing sparse vectors and calculating their dot product. We tested our code by creating two simple vectors and timing how long it took to calculate the dot product of two large sparse vectors.
```

<br/>

123
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/sparse_mul.md
```markdown
1      
2      # Sparse Matrix Multiplication
3      Given two sparse matrices A and B, return the result of AB. You may assume that A's column number is equal to B's row number.
4      
5      ## Example
6      
7      A = 
8      ``` 
9      [
10       [ 1, 0, 0],
11       [-1, 0, 3]
12     ]
13     ```
14     B = 
15     ```
16     [
17       [ 7, 0, 0 ],
18       [ 0, 0, 0 ],
19       [ 0, 0, 1 ]
20     ]
21     ```
22     
23     **AB =** 
24     ```
25     [
26       [  7, 0, 0 ],
27       [-7, 0, 3 ]
28     ]
29     ```
30     
31     ## Solution
32     ### Python solution without table (~156ms)
33     ```python
34     def multiply(self, a, b):
35         """
36         :type A: List[List[int]]
37         :type B: List[List[int]]
38         :rtype: List[List[int]]
39         """
40         if a is None or b is None:
41             return None
42         m, n, l = len(a), len(b[0]), len(b[0])
43         if len(b) != n:
44             raise Exception("A's column number must be equal to B's row number.")
45         c = [[0 for _ in range(l)] for _ in range(m)]
46         for i, row in enumerate(a):
47             for k, eleA in enumerate(row):
48                 if eleA:
49                     for j, eleB in enumerate(b[k]):
50                         if eleB:
51                             c[i][j] += eleA * eleB
52         return c
53     ```
54     
55     ### Python solution with only one table for B (~196ms)
56     ```python
57     def multiply(self, a, b):
58         """
59         :type A: List[List[int]]
60         :type B: List[List[int]]
61         :rtype: List[List[int]]
62         """
63         if a is None or b is None:
64             return None
65         m, n, l = len(a), len(a[0]), len(b[0])
66         if len(b) != n:
67             raise Exception("A's column number must be equal to B's row number.")
68         c = [[0 for _ in range(l)] for _ in range(m)]
69         table_b = {}
70         for k, row in enumerate(b):
71             table_b[k] = {}
72             for j, eleB in enumerate(row):
73                 if eleB:
74                     table_b[k][j] = eleB
75         for i, row in enumerate(a):
76             for k, eleA in enumerate(row):
77                 if eleA:
78                     for j, eleB in table_b[k].iteritems():
79                         c[i][j] += eleA * eleB
80         return c
81     ```
82     
83     ### Python solution with two tables (~196ms)
84     ```python
85     def multiply(self, a, b):
86         """
87         :type A: List[List[int]]
88         :type B: List[List[int]]
89         :rtype: List[List[int]]
90         """
91         if a is None or b is None:
92             return None
93         m, n = len(a), len(b[0])
94         if len(b) != n:
95             raise Exception("A's column number must be equal to B's row number.")
96         l = len(b[0])
97         table_a, table_b = {}, {}
98         for i, row in enumerate(a):
99             for j, ele in enumerate(row):
100                if ele:
101                    if i not in table_a:
102                        table_a[i] = {}
103                    table_a[i][j] = ele
104        for i, row in enumerate(b):
105            for j, ele in enumerate(row):
106                if ele:
107                    if i not in table_b:
108                        table_b[i] = {}
109                    table_b[i][j] = ele
110        c = [[0 for j in range(l)] for i in range(m)]
111        for i in table_a:
112            for k in table_a[i]:
113                if k not in table_b:
114                    continue
115                for j in table_b[k]:
116                    c[i][j] += table_a[i][k] * table_b[k][j]
117        return c
118    ```
```

<br/>

123
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/spiral_traversal.md
```markdown
1      
2      
3      # Spiral Matrix Traversal
4      
5      Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
6      
7      For example,
8      Given the following matrix:
9      
10     ```
11     [
12      [ 1, 2, 3 ],
13      [ 4, 5, 6 ],
14      [ 7, 8, 9 ]
15     ]
16     ```
17     
18     You should return `[1,2,3,6,9,8,7,4,5]`.
19     
20     ## Solution
21     
22     The basic idea is to iterate through the matrix in a spiral pattern, starting from the top-left corner. We can use four variables to keep track of the boundaries of the spiral pattern: `row_begin`, `row_end`, `col_begin`, and `col_end`.
23     
24     ```python
25     def spiral_traversal(matrix):
26         res = []
27         if len(matrix) == 0:
28             return res
29         row_begin = 0
30         row_end = len(matrix) - 1
31         col_begin = 0
32         col_end = len(matrix[0]) - 1
33     ```
34     
35     We can then use a while loop to iterate through the matrix until all elements have been visited.
36     
37     ```python
38     while row_begin <= row_end and col_begin <= col_end:
39         # iterate through the first row from left to right
40         for i in range(col_begin, col_end+1):
41             res.append(matrix[row_begin][i])
42         row_begin += 1
43     ```
44     
45     Now we can iterate through the last column from top to bottom.
46     
47     ```python
48         # iterate through the last column from top to bottom
49         for i in range(row_begin, row_end+1):
50             res.append(matrix[i][col_end])
51         col_end -= 1
52     ```
53     
54     We then need to check if the row_begin is less than or equal to row_end (in case the matrix is a single row) and iterate through the last row from right to left.
55     
56     ```python
57         # iterate through the last row from right to left
58         if row_begin <= row_end:
59             for i in range(col_end, col_begin-1, -1):
60                 res.append(matrix[row_end][i])
61             row_end -= 1
62     ```
63     
64     Finally, we need to check if the col_begin is less than or equal to col_end (in case the matrix is a single column) and iterate through the first column from bottom to top.
65     
66     ```python
67         # iterate through the first column from bottom to top
68         if col_begin <= col_end:
69             for i in range(row_end, row_begin-1, -1):
70                 res.append(matrix[i][col_begin])
71             col_begin += 1
72     
73         return res
74     ```
75     
76     ## Test
77     
78     ```python
79     if __name__ == "__main__":
80         mat = [[1, 2, 3],
81                [4, 5, 6],
82                [7, 8, 9]]
83         print(spiral_traversal(mat))
84     ```
85     
86     Output: `[1, 2, 3, 6, 9, 8, 7, 4, 5]`
```

<br/>

123
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/sudoku_validator.md
```markdown
1      
2      # Valid Solution Function for Sudoku
3      
4      Sudoku is a number-placement puzzle game that requires players to fill a 9x9 grid with numbers in a specific pattern. The goal is to fill all the cells in the grid with numbers from 1 to 9 in such a way that every row, column, and 3x3 subgrid contains all the numbers in the range 1 to 9.
5      
6      In this challenge, we will write a function `valid_solution()` that accepts a 2D array representing a Sudoku board and returns true if it is a valid solution, or false otherwise. The cells of the sudoku board may also contain 0's, which will represent empty cells. Boards containing one or more zeroes are considered to be invalid solutions. The board is always 9 cells by 9 cells, and every cell only contains integers from 0 to 9.
7      
8      ## Solution 1 - Using Dict/Hash-Table
9      
10     The following solution uses a hash-table/dictionary to check if a given board is a valid solution.
11     
12     ```python
13     from collections import defaultdict
14     
15     def valid_solution_hashtable(board):
16         for i in range(len(board)):
17             dict_row = defaultdict(int)
18             dict_col = defaultdict(int)
19             for j in range(len(board[0])):
20                 value_row = board[i][j]
21                 value_col = board[j][i]
22                 if not value_row or value_col == 0:
23                     return False
24                 if value_row in dict_row:
25                     return False
26                 else:
27                     dict_row[value_row] += 1
28     
29                 if value_col in dict_col:
30                     return False
31                 else:
32                     dict_col[value_col] += 1
33     
34         for i in range(3):
35             for j in range(3):
36                 grid_add = 0
37                 for k in range(3):
38                     for l in range(3):
39                         grid_add += board[i * 3 + k][j * 3 + l]
40                 if grid_add != 45:
41                     return False
42         return True
43     ```
44     
45     ## Solution 2 - Without Hash-Table/Dict
46     
47     The following solution uses a standard looping structure to check if a given board is a valid solution.
48     
49     ```python
50     def valid_solution(board):
51         correct = [1, 2, 3, 4, 5, 6, 7, 8, 9]
52         # check rows
53         for row in board:
54             if sorted(row) != correct:
55                 return False
56     
57         # check columns
58         for column in zip(*board):
59             if sorted(column) != correct:
60                 return False
61     
62         # check regions
63         for i in range(3):
64             for j in range(3):
65                 region = []
66                 for line in board[i*3:(i+1)*3]:
67                     region += line[j*3:(j+1)*3]
68     
69                 if sorted(region) != correct:
70                     return False
71     
72         # if everything correct
73         return True
74     ```
75     
76     ## Solution 3 - Using Set
77     
78     The following solution uses a set to check if a given board is a valid solution.
79     
80     ```python
81     def valid_solution_set(board):
82         valid = set(range(1, 10))
83     
84         for row in board:
85             if set(row) != valid:
86                 return False
87     
88         for col in [[row[i] for row in board] for i in range(9)]:
89             if set(col) != valid:
90                 return False
91     
92         for x in range(3):
93             for y in range(3):
94                 if set(sum([row[x*3:(x+1)*3] for row in board[y*3:(y+1)*3]], [])) != valid:
95                     return False
96     
97         return True
98     ```
```

<br/>

Update the md

<!-- empty line --><br/>
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 docs_generated/matrix/sum_sub_squares.md
```markdown
1      
2      
3      # Sum of Sub-Squares of a Matrix
4      
5      Given a square matrix of size `n x n`, find the sum of all its sub-squares of size `k x k` where `k <= n`.
6      
7      ## Algorithm
8      
9      The following algorithm finds the sum of all sub-squares of size `k x k` in a given matrix of size `n x n`.
10     
11     ```
12     # Function to find sum of all
13     # sub-squares of size k x k in a given
14     # square matrix of size n x n
15     def sum_sub_squares(matrix, k):
16         n = len(matrix)
17         result = [[0 for i in range(k)] for j in range(k)]
18     
19         if k > n:
20             return
21         for i in range(n - k + 1):
22             l = 0
23             for j in range(n - k + 1):
24                 sum = 0
25     
26                 # Calculate and print sum of current sub-square
27                 for p in range(i, k + i):
28                     for q in range(j, k + j):
29                         sum += matrix[p][q]
30     
31                 result[i][l] = sum
32                 l += 1
33     
34         return result
35     ```
36     
37     ## Complexity Analysis
38     
39     * Time Complexity: `O(n^2)`, where `n` is the size of the given matrix.
40     
41     * Space Complexity: `O(k^2)`, where `k` is the size of the sub-square.
```

<br/>

# qweqweqwe

*   123
    
*   123
    
*   123
    

<!-- empty line --><br/>

[[sym-mention:(5d526cd4-7a76-4b4c-8060-55cf0304be14|Z2kx9uN)Itay Gersten]]
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 main_test.py
```python
1      import os
2      import openai
3      import glob
4      import ast
5      
6      openai.api_key = "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"
7      # script_path = '/Users/monolite/Documents/GitHub/Meshroom/meshroom/core/cgroup.py'
8      
9      
10     def sum(a, b):
11         return a+b
12     
13     
14     def main():
15         # List all the files in the directory
16         # folder_path = 'algorithms/matrix'
17         folder_path = '/Users/monolite/Documents/GitHub/growth-estimation/Server'
18         # files = os.listdir(folder_path)
19         os.chdir(folder_path)
20         # Iterate over the list of file names
21         for i, file in enumerate(glob.glob('**/*.py', recursive=True)):
22             print(file)
23             if not os.path.isfile(os.path.join(folder_path, file.replace('py', 'md'))) and file.endswith('.py'):
24                 # Open the file
25                 with open(os.path.join(folder_path, file), 'r') as f:
26                     # Read the contents of the file
27                     contents = f.read()
28                     try:
29                         response = openai.Completion.create(
30                             model="text-davinci-003",
31                             prompt=f"create .md file for this script with code snippets:\n`{contents}`",
32                             temperature=0.7,
33                             max_tokens=15000,
34                             top_p=1,
35                             frequency_penalty=0,
36                             presence_penalty=0
37                         )
38                         print(response["choices"][0]["text"])
39                         with open(os.path.join(folder_path, file.replace('py', 'md')), 'w') as output_file:
40                             output_file.write(response["choices"][0]["text"])
41                     except Exception as e:
42                         print(e)
43     
44     
45     if __name__ == '__main__':
46         main()
47         # file_lines = source_code.splitlines()
48         # # Parse the source code into an AST
49         # tree = ast.parse(source_code)
50         #
51         # # Iterate over the AST nodes
52         # for node in ast.walk(tree):
53         #     # Check if the node is a FunctionDef node (i.e., a function definition)
54         #     if isinstance(node, ast.FunctionDef):
55         #         # # Print the function name
56         #         # print(node.name)
57         #         # print('--------------')
58         #         # print('\n'.join(file_lines[node.lineno-1: node.end_lineno]))
59         #         #
60         #         # newline = '\n'
61         #         # # response = openai.Completion.create(
62         #         # #     model="code-davinci-002",
63         #         # #     prompt=f"# Python 3 \n{newline.join(file_lines[node.lineno-1: node.end_lineno])}\n# Explanation of what the code does\n\n#",
64         #         # #     temperature=0,
65         #         # #     max_tokens=64,
66         #         # #     top_p=1.0,
67         #         # #     frequency_penalty=0.0,
68         #         # #     presence_penalty=0.0
69         #         # # )
70         #
71         #         print('--------------')
```

<br/>

<br/>

<br/>

This file was generated by Swimm. [Click here to view it in the app](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBYWxnb3JpdGhtcyUzQSUzQWl0YXlnMjU=/docs/g3vqz).
