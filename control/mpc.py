import numpy as np

class MPC:
    def __init__(self,model, params):
        self.model = model
        self.horizon = params['horizon']
        self.F_arr = params['F_arr']
        self.Wx = params['Wx']
        self.Wf = params['Wf']

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
    
    def optimize(self, current_state, dt):
        best_F = 0
        best_cost = float('inf')

        for F in self.F_arr:
            predicted_state = current_state
            cost = 0

            for _ in range(self.horizon):
                predicted_state = self.f(predicted_state, F, dt)
            cost += (-self.Wx * (predicted_state[0] - current_state[0]) / current_state[0]) + self.Wf * F
            if predicted_state[2] > self.model.capacity:
                cost += 1e5
            if cost < best_cost:
                best_cost = cost
                best_F = F

        return best_F
