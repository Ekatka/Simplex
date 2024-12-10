
import sys
import os
# sys.path.append(os.path.join(os.getcwd(), 'scipy'))
# sys.path.insert(0, '/scipy/my_scipy')
import time
import datetime


from scipy.optimize import linprog

log_file = open('output.log', 'a')

sys.stdout = log_file
print("/n")
print( "Time", datetime.datetime.now())

c = [-3, -2]


A = [
    [-1, 1],
    [1, 0] ,
    [0, 2]
]


b = [1, 3, 2]

x_bounds = (0, None)
y_bounds = (0, None)
start = time.time()

result = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds], method='simplex')

end = time.time()
print("Running time", end - start)

if result.success:
    print("Optimal solution found:")
    print(f"Values of variables: x1 = {result.x[0]}, x2 = {result.x[1]}")
    print(f"Maximum value of z: {-result.fun}")
else:
    print("No solution found.")

log_file.close()
sys.stdout = sys.__stdout__