import numpy as np

class Parameters:
    def __init__(self):
        self.bioreactor_params = {
            'initial_biomass_concentration': 0.1,  # g/L
            'initial_substrate_concentration': 10.0,  # g/L
            'initial_volume': 1.0,  # L
            'initial_waste_concentration': 0.0,  # g/L
            'inhibition_constant': 1 / 0.05,  # g/L
            'substrate_threshold_factor': 1.5, # Substrate threshold factor for inhibition 
            'maximum_specific_growth': 0.4,  # 1/h
            'substrate_saturation_constant': 0.5,  # g/L
            'substrate_decay_rate': 0.05,  # 1/h
            'waste_inhibition_constant': 0.05,  # g/L
            'biomass_yield_coefficient': 0.5,  # g biomass / g substrate
            'waste_production_rate': 0.1,  # g waste / g biomass
            'growth_associated_waste_production_rate': 0.05,  # g waste / g biomass
            'capacity': 2.0,  # L
            'inlet_substrate_concentration': 40.0  # g/L
        }

        self.ekf_params = {
            'Q' :np.array([
                [0.01, 0, 0, 0.01],
                [0, 0.01, 0, 0],
                [0, 0, 0.001, 0],
                [0.01, 0, 0, 0.01]
            ]),  # Process noise covariance
            'R_x' : 0.1**2,
            'R_s' : 0.1**2,
            'R_v' : 0.001**2, # Measurement noise covariance
            'X_interval' : 10, # Measurement interval
            'S_interval' : 20,
            'P' : np.eye(4) * 0.1  # Initial estimation error covariance
        }

        self.grid_params = {
            'grid_size': 50,
            'omega': 2.0,
            'alpha': 0.75,
            'Ds': 2e-3,
            'Dw': 1e-3,
            'intake_source' : [[4, 24, 24, 44 ],[24, 4, 44, 24]]
        }
    
        self.mpc_params = {
            'horizon': 30,
            'F_arr': np.linspace(0, 1, 11),
            'Wx': 1.0,
            'Wf': 0.05
        }
    def get_bioreactor_params(self):
        return self.bioreactor_params

    def get_ekf_params(self):
        return self.ekf_params
    
    def get_grid_params(self):
        return self.grid_params