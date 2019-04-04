'''
This implements the game "Easy 21" as described in the assignment from David Silver's
Reinforcement Learning class and uses SARSA learning to learn a strategy.
The game is a simplified version of Black Jack.  The assignment can be found here:
http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf
'''

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd

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
    
q={}    #action-value function

for s0 in range(1,11):
    for s1 in range(1,22):
        q['['+str(s0)+', '+str(s1)+']hit']=0
        q['['+str(s0)+', '+str(s1)+']stick']=0

e={}    #eligibility trace
for s0 in range(1,11):
    for s1 in range(1,22):
        e['['+str(s0)+', '+str(s1)+']hit']=0
        e['['+str(s0)+', '+str(s1)+']stick']=0

n={}    #counts number of times state has been visited [0] and number of times hit [1]
        #and stick [2] have been selected
for s0 in range(1,11):
    for s1 in range(1,22):
        n['['+str(s0)+', '+str(s1)+']']=0
        n['['+str(s0)+', '+str(s1)+']hit']=0
        n['['+str(s0)+', '+str(s1)+']stick']=0

def alpha(nsa):     #Step size. nsa=number of times action a has been taken in state s
    return 1/nsa

def epsilon(ns, n0=100 ):   #prob. of random action
    return n0/(n0+ns)

def update(q, delta, e, nsa):    #updates action-value function
    return q+e*delta/nsa

def ms_errors(predictions, true):    #This calculates squared errors for predicted values given the true values. (Used for learning curves)
    
    errors=[]
    for i in range(len(true)):
        for j in range(len(true[0])):
            errors.append((true[i][j]-predictions[i][j])**2)

    mse=np.sum(errors)
    return mse

def value(v, dealer, player):
    return v[str([dealer, player])]

mc_values=pd.read_csv(r'C:\Users\walru\OneDrive\Documents\python\easy21MCvalues.csv')
mcv=mc_values.values        #State value function from Monte Carlo (Treated as the "true" values for learning curves)
mcv=np.delete(mcv,0,1)
    
def play(lam):  #Implements a hand of Easy 21 with SARSA(lambda)

    s=initialize()
    for s0 in range(1,11):
        for s1 in range(1,22):
            e['['+str(s0)+', '+str(s1)+']hit']=0
            e['['+str(s0)+', '+str(s1)+']stick']=0

    greedy=np.random.binomial(1, 1-epsilon(n[str(s)]))
    if greedy==1:
        if q[str(s)+'hit']>q[str(s)+'stick']:
            a='hit'
        else:
            a='stick'
    else:
        a=np.random.choice(['hit','stick'])

    s_prime,r=step(s,a)
            
    while s!='terminal':

        n[str(s)]+=1
        n[str(s)+a]+=1
        
        greedy=np.random.binomial(1, 1-epsilon(n[str(s)]))
        if greedy==1:
            if q[str(s)+'hit']>q[str(s)+'stick']:
                a_prime='hit'
            else:
                a_prime='stick'
        else:
            a_prime=np.random.choice(['hit','stick'])
        
        e[str(s)+a]+=1

        if s_prime=='terminal':
            delta=r-q[str(s)+str(a)]
        else:
            delta=q[str(s_prime)+str(a_prime)]-q[str(s)+str(a)]

        for i in q:
            if e[i]!=0:
                q[i]=update(q[i], delta, e[i], n[i])

        for i in e:
            e[i]=e[i]*lam

        s=s_prime
        a=a_prime

        if s!='terminal':
            s_prime,r=step(s,a)

'''
The following plots learning curves for lambda values of 0, 1/2, and 1 using the MC values for comparison.
'''

errors_one=[]

for i in range(200000):
    play(1)

    if i%1000==0:
        
        v={} #value function

        for i in range(1,11):
            for j in range(1,22):
                v['['+str(i)+', '+str(j)+']']=max(q[str([i, j])+'hit'],q[str([i, j])+'stick'])
        Z=np.zeros([21, 10])
        for i in range(21):
            for j in range(10):
                Z[i][j]=value(v, j+1, i+1)
        errors_one.append(ms_errors(Z, mcv))

fig=plt.figure()
ax=plt.axes(projection='3d')

x=np.linspace(1, 10, 10)
y=np.linspace(1, 21, 21)

X, Y=np.meshgrid(x,y)

Z=np.zeros([21, 10])
for i in range(21):
    for j in range(10):
        Z[i][j]=value(v, j+1, i+1)
        
ax.contour3D(X, Y, Z, 100, cmap='binary')
fig.show()

for i in q:
    q[i]=0
errors_zero=[]
for i in range(200000):
    play(0)

    if i%1000==0:
        
        v={} #value function

        for i in range(1,11):
            for j in range(1,22):
                v['['+str(i)+', '+str(j)+']']=max(q[str([i, j])+'hit'],q[str([i, j])+'stick'])
        Z=np.zeros([21, 10])
        for i in range(21):
            for j in range(10):
                Z[i][j]=value(v, j+1, i+1)
        errors_zero.append(ms_errors(Z, mcv))


for i in q:
    q[i]=0
errors_half=[]
for i in range(200000):
    play(0.5)

    if i%1000==0:
        
        v={} #value function

        for i in range(1,11):
            for j in range(1,22):
                v['['+str(i)+', '+str(j)+']']=max(q[str([i, j])+'hit'],q[str([i, j])+'stick'])
        Z=np.zeros([21, 10])
        for i in range(21):
            for j in range(10):
                Z[i][j]=value(v, j+1, i+1)
        errors_half.append(ms_errors(Z, mcv))

learning_curve=plt.figure()
plt.plot(errors_one)
plt.plot(errors_zero)
plt.plot(errors_half)
learning_curve.show()









        
