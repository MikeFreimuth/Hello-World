'''
This implements the game "Easy 21" as described in the assignment from David Silver's
Reinforcement Learning class and uses Monte Carlo learning to learn a strategy.
The game is a simplified version of Black Jack.  The assignment can be found here:
http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf
'''

import numpy as np

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
    
q={}    # initializes action-value function

for s0 in range(1,11):
    for s1 in range(1,22):
        q['['+str(s0)+', '+str(s1)+']hit']=0
        q['['+str(s0)+', '+str(s1)+']stick']=0

n={}    #counts number of times state has been visited [0] and number of times hit [1]
        #and stick [2] have been selected
for s0 in range(1,11):
    for s1 in range(1,22):
        n['['+str(s0)+', '+str(s1)+']']=[0,0,0]

def alpha(nsa):     #step size.  Note nsa is number of times action s has been taken in state a
    return 1/nsa

def epsilon(ns, n0=100 ):  #prob. of random choice
    return n0/(n0+ns)

def update(q, r, alpha):   #updates action-value function
    return ((1-alpha)*q)+(alpha*r)
    
def play():   #implements one hand of the game with MC-control
    actions=[]
    s=initialize()
    
    while s!='terminal':
        n[str(s)][0]+=1
        greedy=np.random.binomial(1, 1-epsilon(n[str(s)][0]))
        if greedy==1:
            if q[str(s)+'hit']>q[str(s)+'stick']:
                a='hit'
            else:
                a='stick'
        else:
            a=np.random.choice(['hit','stick'])
        if a=='hit':
            n[str(s)][1]+=1
            update_action=1
        elif a=='stick':
            n[str(s)][2]+=1
            update_action=2

        actions.append([str(s), str(a)])
        
        s,r=step(s,a)

    for i in actions:
        qi=str(i[0])+str(i[1])
        
        si=str(i[0])
        
        if i[1]=='hit':
            ai=1
        elif i[1]=='stick':
            ai=2
        
        q[qi]=update(q[qi],r,alpha(n[str(si)][ai]))
        
        #print('q value for '+str(qi)+' updated to '+str(q[qi]))
        
for i in range(200000):
    play()

'''
The following code plots the value function determined by running the simulation above
'''

v={} #value function

for i in range(1,11):       #get value of optimal strategy for each state
    for j in range(1,22):
        v['['+str(i)+', '+str(j)+']']=max(q[str([i, j])+'hit'],q[str([i, j])+'stick'])

def value(dealer, player):  #returns the value of the optimal strategy for a given state
    return v[str([dealer, player])]

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fig=plt.figure()
ax=plt.axes(projection='3d')

x=np.linspace(1, 10, 10)
y=np.linspace(1, 21, 21)

X, Y=np.meshgrid(x,y)

Z=np.zeros([21, 10])
for i in range(21):
    for j in range(10):
        Z[i][j]=value(j+1, i+1)
        
ax.contour3D(X, Y, Z, 100, cmap='binary')
fig.show()
output=pd.DataFrame(Z)
output.to_csv(r'C:\Users\walru\OneDrive\Documents\python\easy21MCvalues.csv')
