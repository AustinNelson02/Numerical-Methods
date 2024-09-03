import numpy as np
import copy

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

# We will need to know both the current and past residual throughout 
# the method. They will both start as the same value.
# The projection is the actual step we will take at the end of each iteration.
cur_residual = vector_b - np.matmul(matrix_A,vector_x)
past_residual = copy.deepcopy(cur_residual)
projection = copy.deepcopy(cur_residual)

while step < max_iterations and error > tolerance:
    
    # Performing our one and only matrix-vector product in the method.
    vector_z = np.matmul(matrix_A,projection)
    
    # Determining the step size
    # This step can be performed in one move. It is separated like this
    # so to not have too many characters in one line.
    alpha_1 = np.matmul(np.transpose(past_residual),past_residual)
    alpha_2 = np.matmul(np.transpose(projection),vector_z)
    alpha = alpha_1/alpha_2
    
    # Updating our guess.
    vector_x = vector_x + alpha * projection
    
    # Updating our residual.
    cur_residual = past_residual - alpha * vector_z
    
    # Finding the correct step size.
    beta_1 = np.matmul(np.transpose(cur_residual),cur_residual)
    beta_2 = np.matmul(np.transpose(past_residual),past_residual)
    beta = beta_1/beta_2
    
    # Taking the correct step.
    projection = cur_residual + beta * projection
    
    past_residual = copy.deepcopy(cur_residual)
    
    step = step + 1
    error = np.linalg.norm(cur_residual)
    
# If our method failed, a confirmation of it will print to console.
# Otherwise, we have a solution x and we write that solution to 
# the text file solution_x.txt
if error > tolerance:
    print("The method failed to converge to the solution.")
else:
    with open("solution_x.txt", "w") as txt_file:
        for line in vector_x:
            txt_file.write(str(line[0]) + "\n")
