'''
This implements the game "Easy 21" as described in the assignment from David Silver's
Reinforcement Learning class and uses SARSA with a linear function approximator to learn a strategy.
The game is a simplified version of Black Jack.  The assignment can be found here:
http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf
'''

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd

#The first two functions implement the game

def initialize():
    card=[1,2,3,4,5,6,7,8,9,10]
    initial_state=[card[np.random.randint(0, 10)], card[np.random.randint(0, 10)]]
    return initial_state

def step(s,a):
    #state s is a list containing the dealers card (s[0]) and the player's total (s[1])
    #a is the action ('hit' or 'stick')
    #returns state s' ('terminal' indicates game is finished) and reward r

    card=[1,2,3,4,5,6,7,8,9,10]
    color=[-1,1]

    if s!= 'terminal':
        if a=='hit':
            
            new_color=color[np.random.binomial(1,2/3)]
            new_card=new_color*card[np.random.randint(0, 10)]
            new_total=s[1]+new_card
            #print('new card is ' +str(new_card))
            #print('new total is ' +str(new_total))
            if new_total>21 or new_total<1:
                new_state='terminal'
                r=-1
                #print('you busted!')
            else:
                new_state=[s[0],new_total]
                #print(new_state)
                r=None

        elif a=='stick':
            new_state='terminal'
            dealer_hand=s[0]
            while dealer_hand<17 and dealer_hand>0:
                new_color=color[np.random.binomial(1,2/3)]
                new_card=new_color*card[np.random.randint(0, 10)]
                dealer_hand+=new_card
                #print('dealer total is ' +str(dealer_hand))
            if dealer_hand>21 or dealer_hand<1:
                r=1

            elif dealer_hand<s[1]:
                r=1

            elif dealer_hand==s[1]:
                r=0

            else:
                r=-1

        return (new_state, r)

def value(features, parameters):        #Linear value function
    return float(np.dot(np.transpose(parameters),features))

def update(params, alpha, delta, e):  #updates parameters
    return params+alpha*delta*e

def ms_errors(predictions, true):   #Computes squared errors
    
    errors=[]
    for i in range(len(true)):
        for j in range(len(true[0])):
            errors.append((true[i][j]-predictions[i][j])**2)

    mse=np.sum(errors)
    return mse
    
def features(s,a):      #This defines the feature space as specified in the assignment
    phi=np.zeros([2,3,6])

    dealer=np.zeros((3,1))
    if s[0]>=1 and s[0]<=4:
        dealer[0]=1
    if s[0]>=4 and s[0]<=7:
        dealer[1]=1
    if s[0]>=7 and s[0]<=10:
        dealer[2]=1

    player=np.zeros((6,1))
    if s[1]>=1 and s[1]<=6:
        player[0]=1
    if s[1]>=4 and s[1]<=9:
        player[1]=1
    if s[1]>=7 and s[1]<=12:
        player[2]=1
    if s[1]>=10 and s[1]<=15:
        player[3]=1    
    if s[1]>=13 and s[1]<=18:
        player[4]=1
    if s[1]>=16 and s[1]<=21:
        player[5]=1
        
    state=np.dot(dealer, np.transpose(player))

    if a=='hit':
        phi[0,:,:]=state
    elif a=='stick':
        phi[1,:,:]=state

    phi=np.reshape(phi, [36,1])
    return phi

w=np.zeros([36,1])     #Initialize feature vector with all zeros

mc_values=pd.read_csv(r'C:\Users\walru\OneDrive\Documents\python\easy21MCvalues.csv')
mcv=mc_values.values        #State values from Monte Carlo for computing errors
mcv=np.delete(mcv,0,1)
    
def play(lam, alpha, epsilon, w):   #Implements one hand with SARSA and LFA

    e=np.zeros([36,1])
    
    s=initialize()

    greedy=np.random.binomial(1, 1-epsilon)
    if greedy==1:
        if value(features(s,'hit'),w)>value(features(s,'stick'),w):
            a='hit'
        else:
            a='stick'
    else:
        a=np.random.choice(['hit','stick'])

    s_prime,r=step(s,a)
            
    while s!='terminal':
        
        greedy=np.random.binomial(1, 1-epsilon)
        if greedy==1:
            if value(features(s,'hit'),w)>value(features(s,'stick'),w):
                a_prime='hit'
            else:
                a_prime='stick'
        else:
            a_prime=np.random.choice(['hit','stick'])

        if s_prime=='terminal':
            delta=r-value(features(s,a), w)
        else:
            delta=value(features(s_prime, a_prime), w)-value(features(s,a), w)

        e=lam*e+features(s,a)
        
        w=update(w, alpha, delta, e)
        
        s=s_prime
        a=a_prime

        if s!='terminal':
            s_prime,r=step(s,a)
    return w

'''
The following plots learning curves for lambda of 0, 1/2 and 1.
'''
    
errors_one=[]

for i in range(500000):
    w=play(1, 0.01, 0.05, w)


    if i%1000==0:
        
        Z=np.zeros([21, 10])
        for i in range(21):
            for j in range(10):
                Z[i][j]=max(value(features([j, i], 'hit'), w), value(features([j, i], 'stick'), w))
        errors_one.append(ms_errors(Z, mcv))

fig=plt.figure()
ax=plt.axes(projection='3d')

x=np.linspace(1, 10, 10)
y=np.linspace(1, 21, 21)

X, Y=np.meshgrid(x,y)

Z=np.zeros([21, 10])
for i in range(21):
    for j in range(10):
        Z[i][j]=max(value(features([j, i], 'hit'), w), value(features([j, i], 'stick'), w))
        
ax.contour3D(X, Y, Z, 100, cmap='binary')
fig.show()



w=np.zeros([36,1])
errors_zero=[]
for i in range(500000):
    w=play(0, 0.01, 0.05, w)

    if i%1000==0:
        
        Z=np.zeros([21, 10])
        for i in range(21):
            for j in range(10):
                Z[i][j]=max(value(features([j, i], 'hit'), w), value(features([j, i], 'stick'), w))
        errors_zero.append(ms_errors(Z, mcv))

w=np.zeros([36,1])
errors_half=[]
for i in range(500000):
    w=play(0.5, 0.01, 0.05, w)

    if i%1000==0:
        
        Z=np.zeros([21, 10])
        for i in range(21):
            for j in range(10):
                Z[i][j]=max(value(features([j, i], 'hit'), w), value(features([j, i], 'stick'), w))
        errors_half.append(ms_errors(Z, mcv))

learning_curve=plt.figure()
plt.plot(errors_one)
plt.plot(errors_zero)
plt.plot(errors_half)
learning_curve.show()







        
