from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel

def get_rbf_kernel(length_scale=5.0): # Define the RBF kernel with adjusted parameters Plus length_scale est grand, plus la fonction est "lisse"
    
# le length scale est initialisé à 5.0 et optimisé pour un meilleur ajustement des données
    """RBF kernel avec des paramètres ajustés.""" 
    return RBF(
        length_scale=length_scale,  # met à jour la longueur de l'échelle
        length_scale_bounds=(1e-2, 1e4) # définit les limites de la longueur de l'échelle
    ) + WhiteKernel( # Ajoute un terme de bruit blanc
        noise_level=1e-1,  # définit le niveau de bruit pour meilleur ajustement des données 
        noise_level_bounds=(1e-10, 1e1)  # définit les limites du niveau de bruit 
    )

def get_rational_quadratic_kernel(
    length_scale=2.0, # définit la longueur de l'échelle 
    alpha=0.5,  # définit le paramètre alpha qui contrôle la forme de la fonction de covariance 
    length_scale_bounds=(1e-2, 1e4),  # définit les limites de la longueur de l'échelle
    alpha_bounds=(1e-2, 1e4)  # définit les limites du paramètre alpha
):
    """Rational Quadratic kernel avec des paramètres ajustés."""
    return RationalQuadratic(
        length_scale=length_scale,  
        alpha=alpha,
        length_scale_bounds=length_scale_bounds,
        alpha_bounds=alpha_bounds
    ) + WhiteKernel(
        noise_level=1e-1,   # définit le niveau de bruit pour meilleur ajustement des données
        noise_level_bounds=(1e-10, 1e1)  # définit les limites du niveau de bruit
    )

def get_matern_kernel(
    length_scale=3.0,    # définit la longueur de l'échelle
    nu=1.5,     #  définit le paramètre nu qui contrôle la rugosité de la fonction de covariance 
    length_scale_bounds=(1e-2, 1e4)    # définit les limites de la longueur de l'échelle
):
    """Matern kernel avec des paramètres ajustés."""
    return Matern(
        length_scale=length_scale,
        nu=nu,   
        length_scale_bounds=length_scale_bounds
    ) + WhiteKernel(
        noise_level=1e-1,
        noise_level_bounds=(1e-10, 1e1)
    )

def get_combined_kernel():
    """
    Kernel combiné avec des poids pour équilibrer les contributions.
    """
    k1 = get_rbf_kernel(length_scale=5.0)  # Noyau RBF avec longueur d'échelle ajustée
    k2 = get_rational_quadratic_kernel(
        length_scale=2.0,
        alpha=0.5,
        length_scale_bounds=(1e-2, 1e4),
        alpha_bounds=(1e-2, 1e4)  # Noyau Rationnel Quadratique avec paramètres ajustés
    )
    k3 = get_matern_kernel(length_scale=3.0)  # Noyau Matern avec longueur d'échelle ajustée
    
    # Combinaison pondérée des noyaux
    return 0.4 * k1 + 0.3 * k2 + 0.3 * k3  # Poids équilibrés pour chaque noyau