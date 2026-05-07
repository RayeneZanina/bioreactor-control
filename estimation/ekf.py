import numpy as np

class EKF:
    def __init__(self, model, params):
        self.model = model
        self.f = model.step
        self.h = model.get_measurement
        self.Q = params['Q']
        self.R = params['R']
        self.P = params['P']
        self.state = model.state
    
    def jacobian(self, state, F, dt):
        A = np.zeros((state.shape[0], state.shape[0]))
        epsilon = 1e-5

        f0 = self.f(state, F, dt)
        for i in range(state.shape[0]):
            differential_state = np.copy(state)
            differential_state[i] += epsilon
            f1 = self.f(differential_state, F, dt)
            A[:, i] = (f1 - f0) / epsilon
        return A
    
    def step(self, F ,dt):
        A = self.jacobian(self.state, F, dt)
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  

        x_pred = self.f(self.state, F, dt)
        P_pred = A @ self.P @ A.T + self.Q
        y_pred = self.h(x_pred)
        y = self.h(self.model.state)

        innovation = y - y_pred
        kalman_gain = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + self.R)
        self.state = x_pred + (kalman_gain @ innovation).flatten()
        self.P = (np.eye(self.P.shape[0]) - kalman_gain @ H ) @ P_pred

        return self.state