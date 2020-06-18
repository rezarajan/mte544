import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from math import sin, cos, tan, atan, atan2, sqrt, pi
import random

#***********************************************************************

#  Timestep
dt = 0.1
# Baseline
l = 1.24

# Staring Position
# [x, y, steering, theta]
xhat = np.array([[0.1, 0.2, np.deg2rad(45)]])

# State Covariance Matrix
# P = np.identity(3)
# P = np.zeros((3,3))
# P = np.array([[0.001, 0.001, 0.01], [0.001, 0.001, 0.00001], [0.01, 0.001, 0.0001]])
P = np.array([[0.1, 0.01, 0.01], [0.01, 0.01, 0.0001], [0.01, 0.001, 0.001]])

# Process Noise Covariance Matrix
# Q = np.identity(3)
# Q = np.zeros((3,3))   
Q = np.identity(3)*(0.01**2)
# Q = np.diag(np.array([0.01, 0.01, 0.0001]))*0.01

# Measurement Noise Covariance Matrix
# Part a
R = np.diag(np.array([0.5, 0.5, 1]))
# Part b
# R = np.diag(np.array([0.5, 1]))

# Timesteps
t = np.arange(0,20,dt)
N = len(t)

# Position Matrix
# [  x  , ...]
# [  y  , ...]
# [theta, ...]
pos = np.zeros((3, N))
# Adding Noise
pos[:, 0] = xhat

# Control Matrix
# [    v    , ...]
# [  omega  , ...]
# nonholonomic
u = np.zeros((2, N))

# Encoder Random Error Values
# rnd_pos = np.zeros((3, N))

# Sensor Measurement Values
# Part a
sensor = np.zeros((3, N))
sensor[:,0] = np.array([pos[0,0] + np.random.normal(0, sqrt(R[0,0])), pos[1,0] + np.random.normal(0, sqrt(R[1,1])), atan(pos[1, 0]/pos[0, 0]) + np.random.normal(0, sqrt(R[2,2]))])
# Part b
# sensor = np.zeros((2, N))
# sensor[:,0] = np.array([sqrt(pos[0, 0]**2 + pos[1, 0]**2) + np.random.normal(0, sqrt(R[0,0])), atan(pos[1, 0]/pos[0, 0]) + np.random.normal(0, sqrt(R[1,1]))])

# Sate Matrix
x = np.zeros((3, N))
x[:,0] = xhat


def inverseKinematics(v, w, l):
    v_r = v + w*l/2
    v_l = v - w*l/2

    return v_l, v_r

# EKF Simulation Position and Sensor Matrices
for n in range(0,N-1):
   # Control [v, w]
   u[:,n] = [random.uniform( 0, 0.3 ), random.uniform( -0.5, 0.5 )]
#    u[:,n] = [1*sin(dt*n), 2*cos(dt*n)]
#    u[:,n] = [10*sin(dt*n), 0.01*np.heaviside(n, 1)]

   v_l, v_r = inverseKinematics(u[:,n][0], u[:,n][1], l)

   delta_d = dt*(v_r + v_l)/2
   delta_yaw = dt*(v_r - v_l)/l

   pos[0, n+1] = pos[0, n] + delta_d*cos(pos[2, n] + delta_yaw/2) + np.random.normal(0, sqrt(Q[0,0]))
   pos[1, n+1] = pos[1, n] + delta_d*sin(pos[2, n] + delta_yaw/2) + np.random.normal(0, sqrt(Q[1,1]))
   pos[2, n+1] = pos[2, n] + delta_yaw + np.random.normal(0, sqrt(Q[2,2]))

   # Part a    
   sensor[:, n+1] = np.array([pos[0, n+1] + np.random.normal(0, sqrt(R[0,0])), pos[1, n+1] + np.random.normal(0, sqrt(R[1,1])), atan(pos[1, n+1]/pos[0, n+1]) + np.random.normal(0, sqrt(R[2,2]))])
   # Part b 
#    sensor[:, n+1] = np.array([sqrt(pos[0, n+1]**2 + pos[1, n+1]**2) + np.random.normal(0, sqrt(R[0,0])), atan(pos[1, n+1]/pos[0, n+1]) + np.random.normal(0, sqrt(R[1,1]))])

#***********************************************************************
def ekf(xhat, u, z, Q, R, P, dt, l):

   v_l, v_r = inverseKinematics(u[0], u[1], l)

   delta_d = dt*(v_r + v_l)/2
   delta_yaw = dt*(v_r - v_l)/l

   xhat_k =  np.add(np.array([xhat]), np.array([[delta_d*cos(xhat[2] + delta_yaw/2), delta_d*sin(xhat[2] + delta_yaw/2), delta_yaw]]))
   G = np.array([[1, 0, -delta_d*sin(xhat[2] + delta_yaw/2)], [0, 1, delta_d*cos(xhat[2] + delta_yaw/2)], [0, 0, 1]])
   
   # Linearized Measurement Model Jacobian Matrix
   # Part a
   H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

   # Part b 
#    H = np.array([[xhat[0]/sqrt(xhat[0]**2 + xhat[1]**2), xhat[1]/sqrt(xhat[0]**2 + xhat[1]**2), 0], [-xhat[1]/(xhat[0]**2 + xhat[1]**2), xhat[0]/(xhat[0]**2 + xhat[1]**2), 0]])

   P_predict = np.linalg.multi_dot([G,P,G.T]) + Q
   
   K = np.dot(np.dot(P_predict, H.T),np.linalg.inv((np.linalg.multi_dot([H,P_predict,H.T]) + R)))
   
   # Transposing here is required since xhat was initialized as [1x4] instead of [4x1]
   xhat = np.add(xhat_k, np.dot(K, (np.subtract(np.array([z]).T, np.dot(H,xhat_k.T)))).T)
   P = np.dot((np.subtract(np.identity(3), np.dot(K,H))), P_predict)

   return xhat, P

#***********************************************************************
# EKF Simulation

for n in range(0,N-1):
   # define the matrices as shown in the variable declaration
   x[:, n+1], P = ekf(x[:,n], u[:,n], sensor[:,n], Q, R, P, dt, l)


#***********************************************************************
# Plotting Figures

plotFigure = True
if plotFigure:
   plt.plot(x[0,:], x[1,:], '-b',label="State [x,y]")
   plt.plot(pos[0,:], pos[1,:], '-r',label="Actual [x,y]")
   plt.xlabel('x')
   plt.ylabel('y')

   plt.legend()
   plt.show()