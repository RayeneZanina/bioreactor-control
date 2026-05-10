import numpy as np

class Parameters:
    def __init__(self):
        self.bioreactor_params = {
            'initial_biomass_concentration': 0.1,  # g/L
            'initial_substrate_concentration': 10.0,  # g/L
            'initial_volume': 1.0,  # L
            'initial_waste_concentration': 0.0,  # g/L
            'maximum_specific_growth': 0.4,  # 1/h
            'substrate_saturation_constant': 0.5,  # g/L
            'waste_inhibition_constant': 0.05,  # g/L
            'biomass_yield_coefficient': 0.5,  # g biomass / g substrate
            'waste_production_rate': 0.1,  # g waste / g biomass
            'growth_associated_waste_production_rate': 0.05,  # g waste / g biomass
            'capacity': 5.0,  # L
            'inlet_substrate_concentration': 20.0  # g/L
        }

        self.ekf_params = {
            'Q' : np.diag([0.1, 0.1, 0.01, 0.1]),  # Process noise covariance
            'R' : np.diag([0.1, 0.001]), # Measurement noise covariance
            'P' : np.eye(4) * 0.1  # Initial estimation error covariance
        }

        self.grid_params = {
            'grid_size': 50,
            'omega': 3.0,
            'alpha': 1.0,
            'Ds': 2e-3,
            'Dw': 1e-3,
            'intake_source' : [[0, 24, 24, 49 ],[24, 0, 49, 24]]
        }
    
    def get_bioreactor_params(self):
        return self.bioreactor_params

    def get_ekf_params(self):
        return self.ekf_params
    
    def get_grid_params(self):
        return self.grid_params