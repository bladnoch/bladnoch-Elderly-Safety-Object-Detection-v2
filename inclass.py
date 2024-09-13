import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**3+x**2+1

def function(x):
    h = 0.000000001
    result = (f(x+h)-f(x))/h
    return result

result = []

for i in range(-10,10):
    result.append(function(i))

x=np.linspace(-10,10,100)
result2 = f(x)



print(result)
plt.plot(result)
plt.show()







