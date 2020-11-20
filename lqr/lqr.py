# Pair programming with Eric Rosen
import gym
import control
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def lqr_policy(observation,M,m,l,g):
    # observation: state. 4D in this case.
    x, v, theta, v_theta = observation
    # cost function 

    Q = np.identity(4)
    R = np.identity(1)

    # linearization
    #A = np.identity(4)
    A = [[0,1,0,0],[0,0,-1*(m*g)/M,0],[0,0,0,1],[0,0,((M+m)*g)/(l*M),0]]
    A = np.array(A)

    #B = np.ones((4,1))
    B = [[0],[1/M],[0],[-1/(l*M)]]
    B = np.array(B)

    #K (2-d array) – State feedback gains: ???
    #S (2-d array) – Solution to Riccati equation
    #E (1-d array) – Eigenvalues of the closed loop system
    #print("A:",A.shape)
    #print("B:",B.shape)
    #print("Q:",Q.shape)
    #print("R:",R.shape)
    K, S, E = control.lqr(A,B,Q,R)
    #print("K:",K)
    #print("S:",S)
    #print("E:",E)
    #print("Observation:",observation)
    action = -1*np.dot(K,observation)
    if action >= 0:
        print("action is positive",action)
        return(1)
    else:
        print("action is negative",action)
        return(0)

env = gym.make('CartPole-v1')

M = float(env.masscart)
m = float(env.masspole)
l = float(env.length)
g = float(env.gravity)


frames = []
observation = env.reset()
total_reward = 0

for _ in range(10):
    frames.append(env.render(mode="rgb_array"))
    # env.step() takes an action and returns all we need to know
    observation, reward, done, info = env.step(env.action_space.sample())
    #print(reward)
    total_reward += reward

for _ in range(500):
    frames.append(env.render(mode="rgb_array"))
    # env.step() takes an action and returns all we need to know
    observation, reward, done, info = env.step(lqr_policy(observation,M,m,l,g))
    #print(reward)
    total_reward += reward
env.close()
#save_frames_as_gif(frames)
