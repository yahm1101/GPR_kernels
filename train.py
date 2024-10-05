#•	Ce fichier contient le script principal qui exécute le pipeline complet du projet

#importation des Kernels 

from kernels import get_rbf_kernel, get_linear_kernel, get_matern_kernel
#importation de la class GPR  
from sklearn.gaussian_process import GaussianProcessRegressor


#j'appel le Kernel  que je veux ici :



# Initialiser le modèle GPR avec ce Kernel
gpr = GaussianProcessRegressor(kernel=kernel, alpha=x)  #je met un bruit pour que le model ne surapprend pas (overfitting)


#Entrainer le model pour predir un seul actif

gpr.fit(X_train, y_train)

## Faire des prédictions sur de nouveaux points (X_test)

y_pred, sigma = gpr.predict(X_test, return_std=True) # le std-true c est pour l ecart type que le gpr donne en plus de la prediction



