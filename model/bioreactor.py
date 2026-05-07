import numpy as np

class ideal_Bioreactor:
    def __init__(self, bioreactor_params):
        self.Xi = bioreactor_params['initial_biomass_concentration']  # Biomass concentration (g/L)
        self.Si = bioreactor_params['initial_substrate_concentration']  # Substrate concentration (g/L)
        self.Vi = bioreactor_params['initial_volume']  # Volume of the bioreactor (L)
        self.Wi = bioreactor_params['initial_waste_concentration']  # Waste concentration (g/L)
        self.mu_max = bioreactor_params['maximum_specific_growth']  # Maximum specific growth rate (1/h)
        self.Ks = bioreactor_params['substrate_saturation_constant']  # Substrate saturation constant (g/L)
        self.beta = bioreactor_params['waste_inhibition_constant']  # Waste inhibition constant (g/L)
        self.Y = bioreactor_params['biomass_yield_coefficient']  # Biomass yield coefficient (g biomass/g substrate)
        self.kw = bioreactor_params['waste_production_rate']  # Waste production rate (g waste/g biomass)
        self.kw_growth = bioreactor_params['growth_associated_waste_production_rate']  # Growth-associated waste production rate (g waste/g biomass)
        self.capacity = bioreactor_params['capacity']  # Maximum capacity of the bioreactor (L)
        self.S_in = bioreactor_params['inlet_substrate_concentration']  # Inlet substrate concentration (g/L)

        self.state = np.array([self.Xi, self.Si, self.Vi, self.Wi]) # State vector

    def get_state(self):
        return self.state

    def monod_growth_rate(self, state):
        return self.mu_max * (state[1] / (self.Ks + state[1])) * (1 / (1 + self.beta * state[3]))

    def step(self, state, F, dt):
        intake = min(F * dt, self.capacity - state[2])
        mu = self.monod_growth_rate(state)
        dX_dt = mu * state[0] - F * state[0] / state[2]
        dS_dt = - (mu / self.Y) * state[0] + F * (self.S_in - state[1]) / state[2]
        dV = intake
        dW_dt = self.kw * state[0] + self.kw_growth * mu * state[0] - F * state[3] / state[2]

        new_state = np.array([
            state[0] + dX_dt * dt,
            state[1] + dS_dt * dt,
            state[2] + dV,
            state[3] + dW_dt * dt
        ])

        return new_state

    def update(self,F, dt):
        self.state = self.step(self.state, F, dt)
        return self.get_state()
    
    def get_measurement(self, state):
        return np.array([
            state[0] + np.random.normal(0, 0.01),  # Adding some measurement noise
            state[2] # Assume no noise since volume is easier to measure accurately
        ])


class real_Bioreactor:
    def __init__(self, bioreactor_params, grid_params):

        def initialize_grid(grid_size, total_value, dx):
            grid = np.random.rand(grid_size, grid_size) * 0.1 + 1
            total = np.sum(grid) * dx * dx
            return grid * (total_value / total)

        def velocity_field(grid_size, omega):
            x, y = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
            ux = -omega * (y - 0.5)
            uy = omega * (x - 0.5)
            return ux, uy

        self.Xi = bioreactor_params['initial_biomass_concentration']  # Biomass concentration (g/L)
        self.Si = bioreactor_params['initial_substrate_concentration']  # Substrate concentration (g/L)
        self.Vi = bioreactor_params['initial_volume']  # Volume of the bioreactor (L)
        self.Wi = bioreactor_params['initial_waste_concentration']  # Waste concentration (g/L)
        self.mu_max = bioreactor_params['maximum_specific_growth']  # Maximum specific growth rate (1/h)
        self.Ks = bioreactor_params['substrate_saturation_constant']  # Substrate saturation constant (g/L)
        self.beta = bioreactor_params['waste_inhibition_constant']  # Waste inhibition constant (g/L)
        self.Y = bioreactor_params['biomass_yield_coefficient']  # Biomass yield coefficient (g biomass/g substrate)
        self.kw = bioreactor_params['waste_production_rate']  # Waste production rate (g waste/g biomass)
        self.kw_growth = bioreactor_params['growth_associated_waste_production_rate']  # Growth-associated waste production rate (g waste/g biomass)
        self.capacity = bioreactor_params['capacity']  # Maximum capacity of the bioreactor (L)
        self.S_in = bioreactor_params['inlet_substrate_concentration']  # Inlet substrate concentration (g/L)

        self.state = np.array([self.Xi, self.Si, self.Vi, self.Wi]) # State vector

        self.grid_size = grid_params['grid_size']  # Size of the spatial grid
        self.dx = 1 / self.grid_size # Size of the bioreactor is 1m , so dx = 1 / grid_size
        self.X_grid = initialize_grid(self.grid_size, self.Xi, self.dx)
        self.S_grid = initialize_grid(self.grid_size, self.Si, self.dx)
        self.W_grid = initialize_grid(self.grid_size, self.Wi, self.dx)

        self.omega = grid_params['omega']  # Stirring rate
        self.D = grid_params['D']  # Diffusion coefficient
        self.intake_source = grid_params['intake_source'] # Intake position
        self.ux, self.uy = velocity_field(self.grid_size, self.omega)

    def update(self, F, dt):
        
        def laplacian(grid, dx):
            padded_grid = np.pad(grid, 1, mode='edge')
            return (
                padded_grid[2:, 1:-1] + padded_grid[:-2, 1:-1]
                + padded_grid[1:-1, :-2] + padded_grid[1:-1, 2:]
                - 4 * padded_grid[1:-1, 1:-1]   
            ) / dx**2
        
        def grad_x(grid, dx):
            padded_grid = np.pad(grid, 1, mode='edge')
            return np.where(self.ux > 0, # This is actually really important to avoid oscillations
                            (padded_grid[1:-1, 1:-1] - padded_grid[:-2, 1:-1]) / (dx),
                            (padded_grid[:-2, 1:-1] - padded_grid[1:-1, 1:-1]) / (dx)) 
        
        def grad_y(grid, dx):
            padded_grid = np.pad(grid, 1, mode='edge')
            return np.where(self.uy > 0, 
                            (padded_grid[1:-1, 1:-1] - padded_grid[1:-1, :-2]) / (dx),
                            (padded_grid[1:-1, :-2] - padded_grid[1:-1, 1:-1]) / (dx))
        
        def monod(S_grid, W_grid):
            return self.mu_max * (S_grid / (self.Ks + S_grid)) * (1 / (1 + self.beta * W_grid))

        intake_scalar = min(F * dt, self.capacity - self.state[2])
        intake_grid = np.zeros((self.grid_size, self.grid_size))
        intake_grid[self.intake_source] = intake_scalar

        dX_dt = ( -(self.ux * grad_x(self.X_grid, self.dx) + self.uy * grad_y(self.X_grid, self.dx)) + 
            monod(self.S_grid, self.W_grid) * self.X_grid - 
            F * self.X_grid / self.state[2])
        dS_dt = ( -(self.ux * grad_x(self.S_grid, self.dx) + self.uy * grad_y(self.S_grid, self.dx)) +
            self.D * laplacian(self.S_grid, self.dx) - monod(self.S_grid, self.W_grid) * self.X_grid / self.Y +
            F * (self.S_in - self.S_grid) / self.state[2]
        )
        dW_dt = ( -(self.ux * grad_x(self.W_grid, self.dx) + self.uy * grad_y(self.W_grid, self.dx)) +
                 self.D * laplacian(self.W_grid, self.dx) + self.kw * self.X_grid + 
                 self.kw_growth * monod(self.S_grid, self.W_grid) * self.X_grid -
                 F * self.W_grid / self.state[2]
        )
        dV = intake_scalar
        
        new_X_grid = self.X_grid + dX_dt * dt
        new_S_grid = self.S_grid + dS_dt * dt
        new_W_grid = self.W_grid + dW_dt * dt
        
        new_X = np.sum(new_X_grid) * self.dx * self.dx
        new_S = np.sum(new_S_grid) * self.dx * self.dx
        new_W = np.sum(new_W_grid) * self.dx * self.dx
        new_V = self.state[2] + dV

        self.state = np.array([new_X, new_S, new_V, new_W])
        self.X_grid = new_X_grid
        self.S_grid = new_S_grid
        self.W_grid = new_W_grid

    def get_measurement(self, state):
        return np.array([
            state[0] + np.random.normal(0, 0.01),  # Adding some measurement noise
            state[2] # Assume no noise since volume is easier to measure accurately
        ])