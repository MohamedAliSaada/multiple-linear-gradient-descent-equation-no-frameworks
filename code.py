import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pprint import pprint 
import import_ipynb
import multiple_linear_regression_normal_equation_no_frameworks as OldModule

#load the data and make needed changes 
df = pd.read_csv('ex_2.csv')
col = list(df.columns)
df = df[ [col[-1]]  + [cols for cols in col if cols != col[-1]]   ]
X = df.iloc[:,1:]
X = np.array(X)
Y= df.iloc[:,0]
Y = np.array(Y).reshape(-1,1)

#hyper parameters
alpha=0.01
epochs =1000
m = X.shape[0]

#parameters 
print(f'Y{Y.shape}=X{X.shape} * W(50, 1)  + b(scaler value)')
b = 0
w = np.ones(X.shape[1]).reshape(-1,1)

#cost function calculation 
def cost_clc(m,Y,Y_pred):
    delta = ((Y-Y_pred)**2)*(1/(2*m))
    cost_value = delta.sum()
    return cost_value       

#loop to update for best b and W
cost_history = [] 
for i in range(epochs):
    y_pred = X.dot(w) + b
    cost=cost_clc(m,Y,y_pred)
    cost_history.append(cost)
    if cost <= 10**-3:   #to prevent inf loop 
        break
    #update w and b
    db =  (1/m) * sum(y_pred-Y)
    dw = (1/m) * np.dot(X.T , (y_pred-Y))
    
    b =b -alpha *db 
    w = w-alpha *dw


#check for convergence
plt.figure()
plt.plot(range(len(cost_history)),cost_history,label='cost error',color='r')
e=np.array(cost_history).argmin()
ee=np.array(cost_history).min()
plt.scatter(np.array(cost_history).argmin(),np.array(cost_history).min(),label=f'{ee} @ {e}')
plt.title('check for convergence')
plt.xlabel('epochs')
plt.ylabel('cost_history')
plt.legend()
plt.grid(True)
plt.show()

print('*'*50)
#compare real data with predicted data 
plt.figure()
plt.tight_layout()
plt.scatter(Y,y_pred,label='Predicted data',color='k')
plt.scatter(Y,Y,color='r',label='real data')  
plt.title('compare real data with predicted data')
plt.xlabel('Y')
plt.ylabel('y_pred')
plt.legend()
plt.grid(True)
plt.show()