import numpy as np
import matplotlib.pyplot as plt

'''
This code implements ridge regression and prints learning curves for choosing an appropriate normalization parameter
'''

x=np.loadtxt(r'C:\Users\walru\OneDrive\Documents\python\prostate cancer data.txt', skiprows=1)
y=x[:,-1]
X=x[:,0:-1]
ytrain, ytest = y[:50], y[50:]
Xtrain, Xtest = X[:50], X[50:]
Xbar=np.mean(Xtrain, axis=0)
Xstd=np.std(Xtrain, axis=0)
ybar=np.mean(ytrain)
ytrain=ytrain-ybar
Xtrain=(Xtrain-Xbar)/Xstd
Xtest_standard=(Xtest-Xbar)/Xstd
ytest_standard=ytest-ybar

def ridge(X, y, d2):    #d2=delta squared (regularization parameter)
    theta=np.dot(np.linalg.inv(np.dot(np.transpose(X),X)+d2*np.identity(np.shape(X)[1])),(np.dot(np.transpose(X),y)))
    return theta

thetas=[]
for i in range(-20,40):
    theta=ridge(Xtrain,ytrain,10**(i/10))
    thetas.append(theta)

thetas=np.array(thetas)


theta1=thetas[:,0]
theta2=thetas[:,1]
theta3=thetas[:,2]
theta4=thetas[:,3]
theta5=thetas[:,4]
theta6=thetas[:,5]
theta7=thetas[:,6]
theta8=thetas[:,7]
axis=[]

              
for i in range(60):
    axis.append(i)
'''
#This will produce a regularization path for all the features
fig=plt.plot(axis,theta1,axis,theta2,axis,theta3,axis,theta4,axis,theta5,axis,theta6,axis,theta7,axis,theta8)
plt.show()

'''

def bias(theta,xbar,ybar):
    return ybar-np.dot(np.transpose(xbar),theta)

theta0=[]
for i in range(np.shape(thetas)[0]):
    theta0.append(bias(thetas[i],Xbar,ybar))

def yhat(x,theta,ybar):                   #note that x should be normalized
    return ybar+np.dot(x,theta)

def error(yhat,y):
    errors=yhat-y
    return np.sqrt(np.dot(np.transpose(errors),errors)/np.dot(np.transpose(y),y))

trainingerrors=[]
testerrors=[]

for i in thetas:
    e=error(yhat(Xtrain,i,ybar),ytrain+ybar)
    trainingerrors.append(e)

for i in thetas:
    e=error(yhat(Xtest_standard,i,ybar),ytest)
    testerrors.append(e)

#Learning curves
fig2=plt.plot(axis,trainingerrors,axis,testerrors)
plt.show()
