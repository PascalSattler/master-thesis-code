import numpy as np

position1 = (0,0,0)
position2 = (1,5,0)
position3 = (2,3,0)

list = [position1, position2, position3]
list1 = list[::-1]

concat = np.concatenate((position1, position2), axis=0)

print(concat)

stack = np.stack(list+list1)

print(stack)

x_vals = stack[:,0]
y_vals = stack[:,1]

print(x_vals, y_vals)

print(type(stack))

xmin, xmax = x_vals.min(), x_vals.max()
ymin, ymax = y_vals.min(), y_vals.max()

print(xmax, xmin, ymax, ymin)

config = (3,6,3,6)
n_vertices = np.sum([n for n in config])
print(n_vertices)
print(np.column_stack((np.zeros(n_vertices), np.zeros(n_vertices))))

print(np.vstack((position1, position2)))