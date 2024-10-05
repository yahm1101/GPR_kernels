# •	Dans ce fichier, je définis et je stocke les différents kernels que j'utilise 
# Importer les noyaux depuis scikit-learn
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct

#kernel RBF :

def get_rbf_kernel(length_scale=1.0):
    """
    Renvoie un noyau RBF avec une longueur d'échelle donnée.
    """
    return RBF(length_scale=length_scale)

#kernel lineaire (DotProduct) :
def get_linear_kernel():
    """
    Renvoie un noyau linéaire (produit scalaire).
    """
    return DotProduct()




#kernel matern:


def get_matern_kernel(length_scale=1.0, nu=1.5):
    """
    Renvoie un noyau Matérn avec une longueur d'échelle et un paramètre de lissité (nu).
    """
    return Matern(length_scale=length_scale, nu=nu)

# je pourrais combiner ici les kernels 


def get_combined_kernel():


    combined_kernel = RBF(length_scale=1.0) + DotProduct() + Matern(length_scale=1.0, nu=1.5)