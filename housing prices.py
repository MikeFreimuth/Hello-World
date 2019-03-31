'''
This code builds a random forest and uses it to predict housing prices using data from Kaggle
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\walru\OneDrive\Documents\python\Kaggle Housing Prices\train.csv')
#data.dropna()
vars=list(data.columns.values)
del vars[0]         #this is a list of variable names to use for splitting. 
del vars[-1]        #Dropping the ID and the target variable.
del vars[-4]        #Month sold should probably be converted to something like season but for now I'm just dropping it.

X=data.values                   
X[:,1]=[str(i) for i in X[:,1]]     #Shuffling the data and splitting into train and test sets 
np.random.shuffle(X)                #Using a test set of 200, train of 1260
Xtest=X[:200,1:-1]
Xtest=np.delete(Xtest,-4,1)
Xtrain=X[200:,1:-1]
Xtrain=np.delete(Xtrain,-4,1)
Ytest=X[:200,-1]
Ytest=[np.int32(i) for i in Ytest]
Ytrain=X[200:,-1]

Xtest2=np.delete(Xtest,2,1)
Xtrain2=np.delete(Xtrain,2,1)

def cost(X,Y, split):       #calculates entropy of splitting on variable X at value 'split'
    y1=[]                   
    y2=[]
    if type(split)==str:
        for i in range(len(X)):
            if X[i]==split:
                y1.append(Y[i])
            else:
                y2.append(Y[i])

    else:
        
        for i in range(len(X)):
            if type(X[i])!=str and X[i]>=split:
                y1.append(Y[i])
            else:
                y2.append(Y[i])
                
    l1=len(y1)
    l2=len(y2)
    l=len(Y)
    
    var1=0
    
    if l1>1:
        y1=np.log(np.array(y1))
        y1=np.reshape(y1,[1,l1])
        y1t=np.reshape(y1,[l1,1])      
        cost1=(1/(2*l1))*np.sum(np.square(y1-y1t))  #Entropy equation
    else:
        cost1=0
    
    var2=0
    
    if l2>1:
        y2=np.log(np.array(y2))
        y2=np.reshape(y2,[1,l2])
        y2t=np.reshape(y2,[l2,1]) 
        cost2=(1/(2*l2))*np.sum(np.square(y2-y2t))
    else:
        cost2=0
    
    return cost1+cost2
    
def split(X,Y,n):       #randomly determines n variables to compare for splitting 
    splitvars=[]        #indexes of the variables to split on
    splitX=[]
    Xvals=[]
    while len(splitvars)<n:
        a=np.random.randint(1,len(X[0]))
        if a not in splitvars:
            splitvars.append(a)
            splitX.append(X[...,a])
            Xvals.append(set(X[...,a]))
            
    #print(splitvars)
    
    costs=[]
    splitvals=[]        #these are the values to split on for each variable in splitvars
    for i in range(len(splitvars)):
        newcost=[]
        splits=[]
        for j in Xvals[i]:      #tries splitting on each value and gets the one with the biggest reduction in entropy ('cost')
            newcost.append(cost(splitX[i], Y, j))
            splits.append(j)
        optimal=np.argmin(newcost)
        splitvals.append(splits[optimal])
        costs.append(newcost[optimal])

    #print(splitvars)
    #print(splitvals)
    #print(costs)
   

    best_split=np.argmin(costs)     #compares across the candidate variables at the values determined above to find the best split
    splitvar=splitvars[best_split]
    splitval=splitvals[best_split]
    return splitvar, splitval

class Node:         #Defines a node in a tree
    def __init__(self,x=None,y=None,var=None,split=None):
        self.left=None
        self.right=None
        self.var=var
        self.split=split
        self.x=x
        self.y=y

def build_tree(X,Y,n,m):    #n=maximum depth, m=number of variables to try for each split
    X=np.array(X)
    Y=np.array(Y)
    root=Node(X,Y)
    if n!=0 and len(X)>=6:   
        a,b=split(X,Y,m)
        root.var=a
        root.split=b
        xleft=[]
        xright=[]
        yleft=[]
        yright=[]
        if type(root.split)==str:                       #left is "true" (>=) and right is 'false'
            for i in range(len(X)):
                if X[i][root.var]==root.split:
                    xleft.append(X[i])
                    yleft.append(Y[i])   
                else:
                    xright.append(X[i])
                    yright.append(Y[i])
        else:
            for i in range(len(X)):
                if type(X[i][root.var])!=str and X[i][root.var]>=root.split:
                    xleft.append(X[i])
                    yleft.append(Y[i])
                else:
                    xright.append(X[i])
                    yright.append(Y[i])
        xright=np.array(xright)
        xleft=np.array(xleft)
        n-=1
        if len(yright)!=0 and len(yleft)!=0:
            root.left=build_tree(xleft,yleft,n,m)
            root.right=build_tree(xright,yright,n,m)
        else:
            root.var=None
            root.split=None

    return root
    
def estimate(x,root):       #Gives estimated value of Y for a single tree
    if root.var==None:
        return np.mean(root.y)
    elif type(root.split)==str:
        if x[root.var]==root.split:
            return estimate(x,root.left)
        else:
            return estimate(x,root.right)
    else:
        if type(x[root.var])!=str and x[root.var]>=root.split:
            return estimate(x,root.left)
        else:
            return estimate(x,root.right)
                         
def bag(X,Y,n):     #Getting a "bag" of size n (with replacement) of random observations from X and Y
    m=len(Y)
    bag=np.random.randint(m,size=n)
    bagx=[]
    bagy=[]
    for i in bag:
        bagx.append(X[i])
        bagy.append(Y[i])
    return bagx, bagy

def forest(X,Y,n,d,v,t):                  #n:sample size, d: max depth, v:number of variables to consider for each node, t: number of trees
        roots=[]
        for i in range(t):
            x,y=bag(X,Y,n)
            root=build_tree(x,y,d,v)
            roots.append(root)

        return roots

def evaluate(X,forest):                 #X is np.array, forest is list of roots
    yhat=[]
    for i in X:
        estimates=[]
        for j in forest:
            estimates.append(estimate(i,j))
        yhat.append(np.mean(estimates))
    return yhat

#The following produces two learning curves to compare parameters.

def error(y,yhat):
    return np.sqrt((1/len(y))*np.sum(np.square(np.log(y)-np.log(yhat))))

errors=[]
for i in range(5,300,5):
    f=forest(Xtrain,Ytrain,512,7,7,i)
    yhat=evaluate(Xtest,f)
    errors.append(error(Ytest,yhat))

errors2=[]
for i in range(5,300,5):
    f=forest(Xtrain,Ytrain,512,8,7,i)
    yhat=evaluate(Xtest,f)
    errors2.append(error(Ytest,yhat))
    
plt.plot(errors)
plt.plot(errors2)
plt.show()


