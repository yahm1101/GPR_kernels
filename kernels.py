# •	Dans ce fichier, je définis et je stocke les différents kernels que j'utilise 
# Importer les noyaux depuis scikit-learn


from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel

def get_rbf_kernel(length_scale=1.0, alpha=0.1):
    """
    Renvoie un noyau RBF avec une longueur d'échelle adaptée aux données financières.
    """
    return RBF(length_scale=length_scale) + WhiteKernel(noise_level=alpha)

def get_rational_quadratic_kernel(length_scale=1.0, alpha=1.0):
    """
    Renvoie un noyau Rational Quadratic avec des paramètres adaptés.
    """
    return RationalQuadratic(length_scale=length_scale, alpha=alpha) + WhiteKernel(noise_level=0.1)

def get_matern_kernel(length_scale=3.0, nu=1.5):
    """
    Renvoie un noyau Matérn avec des paramètres adaptés.
    """
    return Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=0.1)

def get_combined_kernel():
    """
    Combine les kernels avec des poids optimisés pour les données financières.
    """
    k1 = 1.0 * RBF(length_scale=3.0)
    k2 = 0.3 * RationalQuadratic(length_scale=3.0, alpha=1.0)
    k3 = 0.5 * Matern(length_scale=3.0, nu=1.5)
    noise = WhiteKernel(noise_level=0.1)
    
    return k1 + k2 + k3 + noise
