import numpy as np
from numpy.linag import eig

A=np.array([[5 ,2], [2,1]])
eigenvalue, eigenvector = eig(A) # w:eigenvalue, v: eigenvector
print (eigenvalue)


