import numpy as np

######################################################################
# Reading in the matrix A and vector b
######################################################################
with open('matrix_A.txt', 'r') as f:
    matrix_A = [[float(num) for num in line.split(' ')] for line in f]

with open('vector_b.txt', 'r') as f:
    vector_b = [[float(num) for num in line.split(' ')] for line in f]

# We convert our matrix and vector into numpy arrays for the sake of
# simplicity when using that format. 
matrix_A = np.array(matrix_A)
vector_b = np.array(vector_b)


######################################################################
# Setting initial parameters for our method.
######################################################################
# The key parameters to consider are the maximum number of iterations
# and the error tolerance. Edit these values to your liking.

max_iterations = 50
tolerance = 1e-8

######################################################################
# Establishing our initial guess.
######################################################################
# This can be set to your liking. We set our initial guess to be the 
# zero vector for the sake of simplicity.

vector_x = np.zeros((len(matrix_A[0]),1))

# With our matrix A, vector b, and our initial parameters, we are ready
# to implement the algorithm.

#First lets set a counter to track the iterations. As well, we will set
# a tracker for our error which we set to be higher than the tolerance.
step = 0
error = 2 * tolerance

######################################################################
# Performing the method.
######################################################################
while step < max_iterations and error > tolerance:
    
    # Computing the residual of the system.
    residual = vector_b - np.matmul(matrix_A,vector_x)
    
    # Determining the step size
    # This step can be performed in one move. It is separated like this
    # so to not have too many characters in one line.
    alpha_1 = np.matmul(np.transpose(residual),residual)
    alpha_2 = np.matmul(np.transpose(residual),np.matmul(matrix_A,residual))
    alpha = alpha_1/alpha_2
    
    # Updating our guess.
    vector_x = vector_x + alpha * residual
    
    # Checking the iteration count and current error to see if 
    # we have converged or failed to arrive at the solution. 
    error = np.linalg.norm(residual)
    step = step + 1
    

# If our method failed, a confirmation of it will print to console.
# Otherwise, we have a solution x and we write that solution to 
# the text file solution_x.txt
if error > tolerance:
    print("The method failed to converge to the solution.")
else:
    with open("solution_x.txt", "w") as txt_file:
        for line in vector_x:
            txt_file.write(str(line[0]) + "\n")