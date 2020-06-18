import matplotlib.pyplot as plt 
import numpy as np

from ekf import ekf



#***********************************************************************

class run_ekf:
    def __init__(self, x=np.zeros((3, 1)), u=np.zeros((2, 1)), sensor = np.zeros((3, 1)), P = np.identity(3), dt = 1):
        self.x = x
        self.u = u
        self.sensor = sensor
        self.P = P
        self.dt = dt
        self.ekf = ekf()


    def calculateEKF(self, ekf):

        # prior, assuming start at (0, 0, 0)
        # [x, y, theta]
        # xhat = np.array([[0, 0, 0]])

        # State Covariance Matrix
        # P = np.identity(3)

        # Process Noise Covariance Matrix
        Q = np.identity(3)*(0.1**2)

        # Measurement Noise Covariance Matrix
        # R = np.identity(3)
        R = np.diag(np.array([0.05, 0.05, 0.1]))

        # Linearized Measurement Model Jacobian Matrix
        H = np.identity(3)

        # State Transition Jacobian
        G = np.array([[1, 0, -1*self.dt*self.u[0]*np.sin(self.x[:,-1][2])], [0, 1, self.dt*self.u[0]*np.cos(self.x[:,-1][2])], [0, 0, 1]])


        # Control Matrix (Velocities and Acceleration)
        # [  x  , ...]
        # [  w  , ...]
        # nonholonomic
        # u = np.zeros((2, 2))
        # u[0:,1] = [0.1, 0]

        # Sensor Measurement Values (Position)
        # Sensor (Position) Matrix
        # [  x  , ...]
        # [  y  , ...]
        # [theta, ...]
        sensor = np.zeros((3, 1))
        sensor[:,0] = self.sensor[:,0]
        # sensor[:,0] = np.add(self.sensor[:,0], [np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(0, 0.1)])
        #sensor[:,1] = np.add(self.sensor[:,1], [np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(0, 0.1)])

        # Sate Matrix
        # x

        #***********************************************************************
        x_pred, P = ekf.ekf(self.x[:,-1], self.u[:,-1], sensor[:,-1], Q, R, G, H, self.P, self.dt)
        X0, Y0, a, b, angle, r_ellipse = ekf.covarianceEllipse(x_pred.reshape(3, -1), P)
        
        # x[:, n+1], P = ekf.ekf(self.x[:,0], u[:,0], sensor[:,1], Q, R, H, P, dt)
        # X0, Y0, _, _, _, r_ellipse = ekf.covarianceEllipse(x[:, n+1].reshape(3, -1), P)

        return x_pred, P, X0, Y0, a, b, angle

    def run(self):
        ekf = self.ekf
        return self.calculateEKF(ekf)