import numpy as np

class EKF:
    def __init__(self, model, params):
        self.model = model
        self.h = model.get_measurement
        self.Q = params['Q']
        self.R = params['R']
        self.P = params['P']
        self.state = model.state

    def f(self, state, F, dt):
        intake = min(F * dt, self.model.capacity - state[2])
        mu = self.model.monod_growth_rate(state)
        dX_dt = mu * state[0] - (intake / dt) * state[0] / state[2]
        dS_dt = - (mu / self.model.Y) * state[0] + (intake / dt) * (self.model.S_in - state[1]) / state[2]
        dV = intake
        dW_dt = self.model.kw * state[0] + self.model.kw_growth * mu * state[0] - (intake / dt) * state[3] / state[2]

        new_state = np.array([
            state[0] + dX_dt * dt,
            state[1] + dS_dt * dt,
            state[2] + dV,
            state[3] + dW_dt * dt
        ])

        return new_state
    
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