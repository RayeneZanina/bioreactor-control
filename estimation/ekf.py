import numpy as np

class EKF:
    def __init__(self, model, params):
        self.model = model
        self.Q = params['Q']
        self.Rx = params['R_x']
        self.Rs = params['R_s']
        self.Rv = params['R_v']
        self.X_interval = params['X_interval']
        self.S_interval = params['S_interval']
        self.P = params['P']
        self.state = model.state

    def f(self, state, F, dt):
        intake = min(F * dt, self.model.capacity - state[2])
        mu = self.model.monod_growth_rate(state)
        dX_dt = mu * state[0] - (intake / dt) * state[0] / state[2]
        dS_dt = - (mu / self.model.Y) * state[0] + (intake / dt) * (self.model.S_in - state[1]) / state[2] - self.model.Sd * state[1]
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
    

    def step(self, F, t ,dt):
        A = self.jacobian(self.state, F, dt)

        x_pred = self.f(self.state, F, dt)

        H_rows, y, y_pred, R_diag = [], [], [], []

        if (t % self.X_interval == 0):
            H_rows.append([1,0,0,0])
            y.append(self.model.state[0] + np.random.normal(0, self.Rx))
            y_pred.append(x_pred[0])
            R_diag.append(self.Rx)

        if (t % self.S_interval == 0):
            H_rows.append([0,1,0,0])
            y.append(self.model.state[1] + np.random.normal(0, self.Rs))
            y_pred.append(x_pred[1])
            R_diag.append(self.Rs)

        H_rows.append([0,0,1,0])
        y.append(self.model.state[2] + np.random.normal(0, self.Rv))
        y_pred.append(x_pred[2])
        R_diag.append(self.Rv)

        H = np.array(H_rows)
        R = np.diag(R_diag)
        y = np.array(y)
        y_pred = np.array(y_pred)

        P_pred = A @ self.P @ A.T + self.Q

        innovation = y - y_pred
        kalman_gain = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        self.state = x_pred + (kalman_gain @ innovation).flatten()
        self.P = (np.eye(self.P.shape[0]) - kalman_gain @ H ) @ P_pred

        return self.state