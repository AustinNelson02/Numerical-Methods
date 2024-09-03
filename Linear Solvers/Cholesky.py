import numpy as np
import copy

######################################################################
# Reading in the matrix A
######################################################################
with open('matrix_A.txt', 'r') as f:
    matrix_A = [[float(num) for num in line.split(' ')] for line in f]
    
# We convert our matrix into numpy arrays for the sake of
# simplicity when using that format. 
matrix_A = np.array(matrix_A)

######################################################################
# Performing the method.
######################################################################

matrix_R = copy.deepcopy(matrix_A)
num_rows = len(matrix_R)

for j in range(0,num_rows):
    matrix_R[j,j] = np.sqrt(matrix_R[j,j])
    
    for i in range(j+1,num_rows):
        matrix_R[i,j] = matrix_R[i,j]/matrix_R[j,j]
    
    for k in range(j+1,num_rows):
        for i in range(k,num_rows):
            matrix_R[i,k] = matrix_R[i,k] - matrix_R[i,j]* matrix_R[k,j]
        
for i in range(0,num_rows-1):
    for j in range(i+1,num_rows):
        matrix_R[i,j] = 0
        
# After our factorization, we print it to the text file
# cholesky_factorization. Due to the nature of the factorization
# we only need to document one of the matrices since the other is 
# its transpose.
with open("cholesky_factorization.txt", "w") as txt_file:
    for line in matrix_R:
        txt_file.write(" ".join(str(element) for element in line) + "\n")
