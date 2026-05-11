# bioreactor-control

This project aims at finding optimal control for a bioreactor by modelling it, simulating real life conditions with incomplete and noisy measurements of the system's state instead of using the true state, recovering estimated state using an extended kalman filter, then finding optimal nutrient inlet profile to maximize biomass and ensure growth of the culture with model predictive control. The idea is to use reduced order MPC by defining a simple model to estimate the state of the system then compute optimal control actions for a high-dimensional more complex system. 

<img width="831" height="228" alt="image" src="https://github.com/user-attachments/assets/84d06b34-bcf6-43af-beaa-ff944ad88ec9" />


I used an extended kalman filter to try to estimate the state of the bioreactor. I first implemented a simple model that assumes perfect homogeneity of the mixture, then implemented a spatial bioreactor model that simulates local concentration gradients. The EKF still uses the first simple model for the state estimation, and we investigate the error it will have when trying to estimate the state of a non-ideal bioreactor.

I then used a MPC to determine the optimal input, using the ideal model that is easier to compute and allows to predict system evolution pretty cheaply.

Main.ipynb covers the information for how the model work with more details as well as a few experiments to try to determine the behaviour of the bioreactor as well as the EKF.

<img width="1015" height="470" alt="image" src="https://github.com/user-attachments/assets/bb558c78-a17a-480e-9a03-5c84b8712873" />
The EKF is able to estimate the state and remove the measurement noise pretty well, as well as estimate variables that are not measured like the waste.
<img width="584" height="435" alt="image" src="https://github.com/user-attachments/assets/11271968-2cf0-4fd0-9b42-2dc54c467827" />
This holds for the real bioreactor simulation up to a certain point. As long as the mixture is stirred enough and the inlet is placed strategically to allow for diffusion of the substrate, the EKF shows similar error in the estimation, which is insignificant.
<img width="584" height="435" alt="image" src="https://github.com/user-attachments/assets/fc27df8e-92f0-46f2-b501-83be75eb5ed7" />
The MPC is then able to determine optimal feed for the system. I tried a few different feed profiles, and the MPC feed outperformed all of them.
<img width="572" height="455" alt="image" src="https://github.com/user-attachments/assets/cdc324cd-8f77-4f87-8035-fa2b104864ba" />







