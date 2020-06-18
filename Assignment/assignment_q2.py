import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from math import sin, cos, tan, atan, atan2, sqrt, pi


#***********************************************************************

dt = 0.1
l = 1

# prior, assuming start at (0, 0, 0)
# [x, y, steering, theta]
xhat = np.array([[0, 0.1, np.deg2rad(2), np.deg2rad(30)]])

# State Covariance Matrix
# Note: High impact on EKF accuracy - affects path accuracy

# Choice 1
#  Worst Case - high impact, produces an extremely offset EKF
# P = np.identity(4)
# Choice 2
# Bad Case - high impact on EKF, produces filter which is noisy, offset and skewed
# P = np.zeros((4,4))
# Choice 3
# Realistic Case - high impact on EKF, produces filter which is almost an exact match to the state
# Test (i) - worse
# P = np.identity(4)*0.001
# Test (ii) - better
# P = np.identity(4)*0.0001
# Test (iii) - worse
P = np.identity(4)*0.00001
# baseline as Test (iii)

# Refining
# Test (i) - x, y as larger covariance - out of line with path mostly on the y-variable
# P = np.array([[0.1, 0.1, 0.0001, 0.0001], [0.1, 0.1, 0.0001, 0.0001], [0.1, 0.1, 0.0001, 0.0001], [0.1, 0.1, 0.0001, 0.0001]])
# Test (ii) - steering, heading as larger covariance - closer to path's shape, but curvature is off
# P = np.array([[0.0001, 0.0001, 0.1, 0.1], [0.0001, 0.0001, 0.1, 0.1], [0.0001, 0.0001, 0.1, 0.1], [0.0001, 0.0001, 0.1, 0.1]])
# Test (iii) - x & heading as larger covariance - closer to path's shape, but curvature is off
# P = np.array([[0.1, 0.0001, 0.0001, 0.1], [0.01, 0.0001, 0.0001, 0.1], [0.1, 0.0001, 0.0001, 0.1], [0.1, 0.0001, 0.0001, 0.1]])
# P = np.array([[0.001, 0.001, 0.01, 0.1], [0.001, 0.001, 0.01, 0.1], [0.001, 0.001, 0.01, 0.1], [0.001, 0.001, 0.01, 0.1]])
# P = np.array([[0.001, 0.1, 0.0001, 0.0001], [0.001, 0.1, 0.0001, 0.0001], [0.001, 0.1, 0.0001, 0.0001], [0.001, 0.1, 0.0001, 0.0001]])

# P = np.array([[0.1, 0.01, 0.01, 0.0001], [0.01, 0.01, 0.0001, 0.00001], [0.01, 0.001, 0.001, 0.00001], [0.0001, 0.0001, 0.00001, 0.00001]])
# Increasing x covariance in first row - very sensitive; worse
# P = np.array([[0.5, 0.01, 0.01, 0.0001], [0.01, 0.01, 0.0001, 0.00001], [0.01, 0.001, 0.001, 0.00001], [0.0001, 0.0001, 0.00001, 0.00001]])



# Using Test (i) as the best selection for P



#***********************************************************************

# Process Noise Covariance Matrix
# Note: High impact on EKF accuracy - affects path noise

# Choice 1
#  Worst Case - high impact, produces a very noisy path
# Q = np.identity(4)
# Choice 2
# Q = np.zeros((4,4))
# Choice 3 (Realistic Case)
# Test (i) - Noisy, inaccurate path
# Q = np.identity(4)*0.01 
# Test (ii) - Noisy, inaccurate path, better
# Q = np.identity(4)*0.0001 
# Test (iii) - Noisy, inaccurate path, closer shape, better
# Q = np.diag(np.array([0.01, 0.01, 0.0001, 0.0001]))
# Test (iii) - Much closer shape, best
Q = np.diag(np.array([0.0001, 0.0001, 0.000001, 0.000001]))*0.01

# Using Test (iii) as the best selection for Q

#***********************************************************************

# Measurement Noise Covariance Matrix
# Fixed
R = np.diag(np.array([0.1, 0.3]))

# Timesteps
t = np.arange(0,10,dt)
N = len(t)

# Position Matrix
# [  x  , ...]
# [  y  , ...]
# [steering, ...]
# [theta, ...]
pos = np.zeros((4, N))
# Adding Noise
pos[:, 0] = xhat

# Control Matrix
# [  directional velocity  , ...]
# [   steering velocity    , ...]
# nonholonomic
u = np.zeros((2, N))
u[0:,1] = [10*sin(dt), 0.01]

# Sensor Measurement Values
sensor = np.zeros((2, N))
sensor[:,0] = np.array([sqrt(pos[0, 0]**2 + pos[1, 0]**2) + np.random.normal(0, sqrt(R[0,0])), atan(pos[1, 0]/pos[0, 0]) + np.random.normal(0, sqrt(R[1,1]))])

# Sate Matrix
x = np.zeros((4, N))
x[:,0] = xhat

# EKF Simulation Position and Sensor Matrices
for n in range(0,N-1):
   # Constant Velocity and Heading
   u[:,n] = [10*sin(dt*n), 0.01*np.heaviside(n, 1)]

   # pos[0, n+1] = pos[0, n] + dt*u[0, n]*cos(pos[3, n]) + np.random.normal(0, sqrt(Q[0,0]))
   # pos[1, n+1] = pos[1, n] + dt*u[0, n]*sin(pos[3, n]) + np.random.normal(0, sqrt(Q[1,1]))
   # pos[2, n+1] = pos[2, n] + dt*u[1, n] + np.random.normal(0, sqrt(Q[2,2]))
   # pos[3, n+1] = pos[3, n] + dt*u[0, n]*tan(pos[2, n])/l + np.random.normal(0, sqrt(Q[3,3]))
   pos[0, n+1] = pos[0, n] + dt*u[0, n]*cos(pos[3, n])
   pos[1, n+1] = pos[1, n] + dt*u[0, n]*sin(pos[3, n])
   pos[2, n+1] = pos[2, n] + dt*u[1, n]
   pos[3, n+1] = pos[3, n] + dt*u[0, n]*tan(pos[2, n])/l
   
   sensor[:, n+1] = np.array([sqrt(pos[0, n+1]**2 + pos[1, n+1]**2) + np.random.normal(0, sqrt(R[0,0])), atan(pos[1, n+1]/pos[0, n+1]) + np.random.normal(0, sqrt(R[1,1]))])

#***********************************************************************
def ekf(xhat, u, z, Q, R, P, dt, l):
   xhat_k =  np.add(np.array([xhat]), np.array([[dt*u[0]*cos(xhat[3]), dt*u[0]*sin(xhat[3]), dt*u[1], dt*u[0]*tan(xhat[2])/l]]))
   G = np.array([[1, 0, 0, -dt*u[0]*sin(xhat[3])], [0, 1, 0, dt*u[0]*cos(xhat[3])], [0, 0, 1, 0], [0, 0, dt*u[0]*(1/(cos(xhat[2])**2))/l, 1]])
   
   # Linearized Measurement Model Jacobian Matrix
   H = np.array([[xhat[0]/sqrt(xhat[0]**2 + xhat[1]**2), xhat[1]/sqrt(xhat[0]**2 + xhat[1]**2), 0, 0], [-xhat[1]/(xhat[0]**2 + xhat[1]**2), xhat[0]/(xhat[0]**2 + xhat[1]**2), 0, 0]])

   P_predict = np.linalg.multi_dot([G,P,G.T]) + Q
   
   K = np.dot(np.dot(P_predict, H.T),np.linalg.inv((np.linalg.multi_dot([H,P_predict,H.T]) + R)))
   
   # Transposing here is required since xhat was initialized as [1x4] instead of [4x1]
   xhat = np.add(xhat_k, np.dot(K, (np.subtract(np.array([z]).T, np.dot(H,xhat_k.T)))).T)
   P = np.dot((np.subtract(np.identity(4), np.dot(K,H))), P_predict)

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
   plt.subplot(4,1,1)
   plt.plot(t, pos[0,:], label="Actual")
   plt.plot(t, x[0,:], label="Predicted")
   plt.xlabel('Time (s)')
   plt.ylabel('Position [X]')

   plt.subplot(4,1,2)
   plt.plot(t, pos[1,:], label="Actual")
   plt.plot(t, x[1,:], label="Predicted")
   plt.xlabel('Time (s)')
   plt.ylabel('Position [Y]')

   plt.subplot(4,1,3)
   plt.plot(t, np.rad2deg(pos[2,:]), label="Actual")
   plt.plot(t, np.rad2deg(x[2,:]), label="Predicted")
   plt.xlabel('Time (s)')
   plt.ylabel('Steering Angle [Delta]')

   plt.subplot(4,1,4)
   plt.plot(t, np.rad2deg(pos[3,:]), label="Actual")
   plt.plot(t, np.rad2deg(x[3,:]), label="Predicted")
   plt.xlabel('Time (s)')
   plt.ylabel('Heading [Theta]')

   plt.legend()
   plt.show()

if plotFigure:
   plt.figure(2)
#    plt.plot(r_ellipse[:,0] + X0, r_ellipse[:,1] + Y0, label="Ellipse")
   plt.plot(x[0,:], x[1,:], '-b',label="State EKF[x,y]")
   plt.plot(pos[0,:], pos[1,:], '-r',label="Actual [x,y]")
   plt.xlabel('x')
   plt.ylabel('y')

   plt.legend()
   plt.show()