# mat_calculate
calculate matrix

with BUG

mainpage https://github.com/MaXinglong/playground/tree/master/matrix


# Usage

## 1. declare a dense matrix

### without Init matrix's rows or cols
```
DenseMatrix A;
```
### declare with rows, cols and fill the matrix use the third value
```
DenseMatrix A(3,4,5);
```
### declare matrix use other Matrix, this with construct a new object
```
DenseMatrix B(A);
```
### this will move A to B, remember do NOT use A again
```
DenseMatrix A(3,4,5);
DenseMatrix B;
B = A;
```
### use array initialize Matrix
```
double p[3][4] = {{2 3 1 5},{1 6 4 2},{5 5 9 5}};
DenseMatrix A(p,3,4);
```
### use 1 dim array declare vector
```
// col vector
double p[5] = {1,3,4,5,1};
DenseMatrix A(p,5,1);
// row vector
DenseMatrix B(p,5,0);
```
## 2. calculate
### add
```
DenseMatrix C = A + B;
C = A + 10;
C.add(B);
```
### sub
```
DenseMatrix C = A - B;
C = A - 10;
C.sub(10);
C.sub(B);
```
### mul
```
DenseMatrix A(4,5,20);
DenseMatrix B(5,6,30);
DenseMatrix C(4,5,22);
DenseMatrix D;
D = A * C.T;
```
### Elementary transformation of matrix
```
DenseMatrix A;
A = DenseMatrix::eye(5);
cout << A;
A.Rix_k(1,3.1,2);
cout << A;
```
### determinant (it's difficult to fill a matrix, later will load a file from .csv, and can compare the result with other software like matlab or python...)
```
double_t p2[5][5] = {{17,24,1,8,15},{23,5,7,14,16},{4,6,13,20,22},{10,12,19,21,3},{11,18,25,2,9}};
DenseMatrix kp2(5,5);
kp2.setRow(0,p2[0],5);
kp2.setRow(1,p2[1],5);
kp2.setRow(2,p2[2],5);
kp2.setRow(3,p2[3],5);
kp2.setRow(4,p2[4],5);
std::cout << kp2;
std::cout << kp2.det()/100000.0 << std::endl;
```
### exp, sum, log, pow
```
DenseMatrix A;
A = B.log();
A = B.exp();
A = A.pow(2);
A = B.log()*B.exp(); //...
```
## except broadcasting!
## and now can use this lib to implement a neural network!
## excited~
