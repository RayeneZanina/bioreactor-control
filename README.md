# bioreactor-control

This project aims at finding optimal control for a bioreactor by modelling it, simulating real life conditions with incomplete and noisy measurements of the system's state instead of using the true state, recovering estimated state using an extended kalman filter, then finding optimal nutrient inlet profile to maximize biomass and ensure growth of the culture. 

I used an extended kalman filter to try to estimate the state of the bioreactor. I first implemented a simple model that assumes perfect homogeneity of the mixture, then implemented a spatial bioreactor model that simulates local concentration gradients. The EKF still uses the first simple model for the state estimation, and we investigate the error it will have when trying to estimate the state of a non-ideal bioreactor.

Main.ipynb covers the information for how the model work with more details as well as a few experiments to try to determine the behaviour of the bioreactor as well as the EKF.

<img width="1028" height="470" alt="image" src="https://github.com/user-attachments/assets/7d7b9258-2a16-4e5f-b222-c2b4a10addae" />
The EKF showed minimal error for the measured state properties, although the error in waste which is not measured seems to increase over time
<img width="594" height="455" alt="image" src="https://github.com/user-attachments/assets/77bbb96f-0016-4ca8-b206-22a5aa58861c" />
The EKF managed to effectively halve the noise from the measurement, as well as give continuous state estimation.
<img width="576" height="435" alt="image" src="https://github.com/user-attachments/assets/cfd03ae2-17bc-476a-9423-fe74083835ef" />
The non-ideal bioreactor behaves differently from the ideal one due to some local starvation.
<img width="1028" height="470" alt="image" src="https://github.com/user-attachments/assets/388c748e-69cd-4dd2-82db-40d24f0a9dac" />
The EKF is still able to accurately estimate the state, up to a certain extent. If the non-ideal bioreactor doesn't have strong enough stirring and mixing of the solution, the EKF shows a jittery noisy signal and significant error.
<img width="1015" height="470" alt="image" src="https://github.com/user-attachments/assets/c411997f-8dae-40eb-8c5e-8268a8a49430" />



