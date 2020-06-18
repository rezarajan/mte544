import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from math import sin, cos, tan, atan, atan2, sqrt, pi
import random

#***********************************************************************

#  Timestep
dt = 0.1
# Baseline
l = 0.45

# Staring Position
# [x, y, steering, theta]
xhat = np.array([[0.1, 0.2, np.deg2rad(45)]])

# Timesteps
t = np.arange(0,50,dt)
N = len(t)

# Position Matrix
# [  x  , ...]
# [  y  , ...]
# [theta, ...]
pos = np.zeros((3, N))
# Adding Noise
pos[:, 0] = xhat

# Control Matrix
# [  omega_l  , ...]
# [  omega_r  , ...]
# nonholonomic
u = np.zeros((2, N))
# u[0:,1] = [10*sin(dt), 20*sin(dt)]
# u[0:,1] = [10, 0]

# Encoder Random Error Values
rnd_pos = np.zeros((3, N))

# Sensor Measurement Values
# sensor = np.zeros((2, N))
# sensor[:,0] = np.array([sqrt(pos[0, 0]**2 + pos[1, 0]**2), atan(pos[1, 0]/pos[0, 0])])

# Sate Matrix
# x = np.zeros((3, N))
# x[:,0] = xhat


# EKF Simulation Position and Sensor Matrices
for n in range(0,N-1):
   # Right and Left Wheel Angular Velocities
   u[:,n] = [1*sin(dt*n), 2*cos(dt*n)]
#    u[:,n] = [1, 2]

   delta_d = dt*(u[:,n][1] + u[:,n][0])/2
   delta_yaw = dt*(u[:,n][1] - u[:,n][0])/l

   pos[0, n+1] = pos[0, n] + delta_d*cos(pos[2, n] + delta_yaw/2)
   pos[1, n+1] = pos[1, n] + delta_d*sin(pos[2, n] + delta_yaw/2)
   pos[2, n+1] = pos[2, n] + delta_yaw

# Random Error
def rnderrEncoder(err_pct_l = 2, err_pct_r = 2):
    rnd_pos[:, 0] = pos[:, 0]
    for n in range(0,N-1):
        # Generate random errors
        r = random.randint(0, 1)

        if(r):
            mult_r = (100-err_pct_r)/100
        else:
            mult_r = 1

        if(l):
            mult_l = (100-err_pct_l)/100
        else:
            mult_l = 1

        delta_d = dt*(u[:,n][1]*mult_r + u[:,n][0]*mult_l)/2
        delta_yaw = dt*(u[:,n][1]*mult_r - u[:,n][0]*mult_l)/l    

        rnd_pos[0,n+1] = rnd_pos[0, n] + delta_d*cos(rnd_pos[2, n] + delta_yaw/2)
        rnd_pos[1,n+1] = rnd_pos[1, n] + delta_d*sin(rnd_pos[2, n] + delta_yaw/2)
        rnd_pos[2, n+1] = rnd_pos[2, n] + delta_yaw

    return rnd_pos


#***********************************************************************
# Plotting Figures

plotFigure = True
if plotFigure:
   plt.figure(1)
   plt.plot(pos[0,:], pos[1,:], label="Actual")
   plt.xlabel('Position [X]')
   plt.ylabel('Position [Y]')
   plt.legend()
   plt.show()

   rnd_pos = rnderrEncoder(2, 2)
   # RMSE
   print("RMSE [L: 2% | R: 2%]: {}".format(np.sqrt(np.mean((rnd_pos-pos)**2))))
   plt.figure(2)
   plt.plot(pos[0,:], pos[1,:], label="Actual")
   plt.plot(rnd_pos[0,:], rnd_pos[1,:], label="Random Error [2%L | 2%R]")
   plt.xlabel('Position [X]')
   plt.ylabel('Position [Y]')
   plt.legend()
   plt.show()

   rnd_pos = rnderrEncoder(3, 2)
   # RMSE
   print("RMSE [L: 3% | R: 2%]: {}".format(np.sqrt(np.mean((rnd_pos-pos)**2))))
   plt.figure(3)
   plt.plot(pos[0,:], pos[1,:], label="Actual")
   plt.plot(rnd_pos[0,:], rnd_pos[1,:], label="Random Error [3%L | 2%R]")
   plt.xlabel('Position [X]')
   plt.ylabel('Position [Y]')

   plt.legend()
   plt.show()
