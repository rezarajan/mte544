# Updating at 1Hz
import numpy as np
from scipy import stats
from math import sin, cos, atan2, sqrt, pi

#***********************************************************************

class ekf:
    def __init__(self, x=np.zeros((3, 1)), xhat = np.array([[0, 0, 0]]).T, pos=np.zeros((3, 1)), u=np.zeros((2, 1)), sensor = np.zeros((3, 1)),\
        dt=1, P=np.identity(3), Q=np.identity(3), R=np.identity(3), H=np.identity(3), G=np.zeros((3,1))):
        # Sate Matrix
        self.x=x

        # prior, assuming start at (0, 0, 0)
        # [x, y, theta]
        self.xhat=xhat

        # Position Matrix
        # [  x  , ...]
        # [  y  , ...]
        # [theta, ...]
        self.pos=pos

        # Control Matrix
        # [  x  , ...]
        # [  y  , ...]
        # nonholonomic
        self.u=u

        # Sensor Measurement Values
        self.sensor=sensor

        # Timestep
        self.dt=dt

        # State Covariance Matrix
        self.P=P
        # Process Noise Covariance Matrix
        self.Q=Q
        # Measurement Noise Covariance Matrix
        self.R=R
        # Linearized Motion Model Jacobian Matrix
        # self.G = np.array([[1, 0, -1*dt*u[0]*sin(self.xhat[2])], [0, 1, dt*u[0]*cos(self.xhat[2])], [0, 0, 1]])
        # Linearized Measurement Model Jacobian Matrix
        self.H=H

    #***********************************************************************
    def ekf(self, xhat = None, u = None, z = None, Q = None, R = None, G = None, H = None, P = None, dt = None):
        if xhat is None: xhat = self.xhat
        if u is None: u = self.u
        if z is None: z = self.sensor
        if Q is None: Q = self.Q
        if R is None: R = self.R
        if G is None: G = self.G
        if H is None: H = self.H
        if P is None: P = self.P
        if dt is None: dt = self.dt

        xhat_k =  np.add(np.array([xhat]), np.array([[dt*u[0]*cos(xhat[2]), dt*u[0]*sin(xhat[2]), dt*u[1]]]))
        # G = np.array([[1, 0, -1*dt*u[0]*sin(xhat[2])], [0, 1, dt*u[0]*cos(xhat[2])], [0, 0, 1]])
        
        P_predict = np.add(np.linalg.multi_dot([G,P,G.T]), Q)
        
        K = np.dot(np.dot(P_predict, H.T),np.linalg.inv((np.add(np.linalg.multi_dot([H,P_predict,H.T]), R))))
        # Transposing here is required since xhat was initialized as [1x3] instead of [3x1]
        xhat = np.add(xhat_k, np.dot(K, (np.subtract(np.array([z]).T, np.dot(H,xhat_k.T)))).T)
        P = np.dot((np.subtract(np.identity(P_predict.shape[1]), np.dot(K,H))), P_predict)
        return xhat, P

    #***********************************************************************
    def covarianceEllipse(self, X=np.zeros(2), P=np.eye(2)):
        # Only x and y covariances
        cov = P[0:2, 0:2]

        eigenval, eigenvec  = np.linalg.eig(cov)
        eigenval = eigenval[:2]
        eigenvec = eigenvec[:,:2]

        largest_eigenval = np.amax(eigenval)
        largest_eigenvec_ind_c = np.argmax(eigenval == largest_eigenval)
        largest_eigenvec = eigenvec[:, largest_eigenvec_ind_c]

        smallest_eigenval = np.amin(eigenval)
        smallest_eigenvec_ind_c = np.argmax(eigenval == smallest_eigenval)
        smallest_eigenvec = eigenvec[:, smallest_eigenvec_ind_c]

        angle = atan2(largest_eigenvec[1],largest_eigenvec[0])
        if angle < 0 : angle += 2*pi

        chisquare_val = stats.chi2.ppf(0.95, 2)
        theta_grid = np.linspace(0, 2*pi, num=100)
        X0 = X[0,-1]
        Y0 = X[1,-1]
        a = sqrt(chisquare_val*largest_eigenval)
        b = sqrt(chisquare_val*smallest_eigenval)
        ellispe_x_r = a*np.cos(theta_grid)
        ellipse_y_r = b*np.sin(theta_grid)
        R = np.array([[cos(angle), sin(angle)], [-sin(angle), cos(angle)]])
        ellipse_array = np.array([ellispe_x_r, ellipse_y_r])

        r_ellipse = R.dot(ellipse_array)

        return X0, Y0, a, b, angle, r_ellipse
