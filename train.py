import os # Pour créer des répertoires 
import pandas as pd  # Pour manipuler les données 
import yfinance as yf  # Pour télécharger les données historiques
import numpy as np  # Pour les calculs numériques  
from sklearn.preprocessing import StandardScaler  # Pour normaliser les données 
from sklearn.gaussian_process import GaussianProcessRegressor  # Pour entraîner le modèle GPR 
from visualisation import plot_predictions_with_uncertainty  # Pour visualiser les prédictions  
from kernels import get_rbf_kernel, get_matern_kernel, get_rational_quadratic_kernel, get_combined_kernel  # Pour obtenir les noyaux 
import matplotlib.pyplot as plt  # Pour les graphiques 
from scipy.optimize import minimize # Pour l'optimisation 
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error # Pour les métriques de performance

 # Fonction pour calculer les métriques de performance 
def calculate_metrics(y_true, y_pred): 
    """
    Calcule les métriques de performance.
    """
    mse = mean_squared_error(y_true, y_pred)  # Erreur quadratique moyenne 
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Racine carrée de l'erreur quadratique moyenne 
    mae = mean_absolute_error(y_true, y_pred)  # Erreur absolue moyenne  
    mape = mean_absolute_percentage_error(y_true, y_pred)  # Erreur absolue moyenne en pourcentage 
    return mse, rmse, mae, mape



# Optimiseur personnalisé pour le modèle GPR avec L-BFGS-B et un nombre maximal d'itérations  

def custom_optimizer(obj_func, initial_theta, bounds):
    """
    Optimiseur personnalisé utilisant scipy.optimize.minimize avec L-BFGS-B
    et un nombre maximal d'itérations défini.
    """
    
    result = minimize(              # Minimise la fonction objective 
        fun=obj_func,               # Fonction objective 
        x0=initial_theta,           # Initialisation des paramètres  
        method='L-BFGS-B',          # Méthode d'optimisation 
        jac=True,                   # Indique que la fonction objective retourne également le gradient
        bounds=bounds,              # Limites des paramètres
        options={'maxiter': 1000}   # Définir le nombre maximal d'itérations 
    )
    return result.x, result.fun     # Retourne les paramètres optimaux et la valeur de la fonction objective

# Classe pour entraîner le modèle GPR avec différents noyaux et évaluer les performances
class FinancialML:
    def __init__(self, ticker: str, start_date: str, end_date: str, window_size: int = 5, test_size: float = 0.2):
        """
        Initialise la classe avec les paramètres de base.
        """
        self.ticker = ticker            # Symbole de l'actif 
        self.start_date = start_date    # Date de début 
        self.end_date = end_date        # Date de fin 
        self.window_size = window_size  # Taille de la fenêtre 
        self.test_size = test_size      # Taille de l'ensemble de test 
        
        # Initialisation des attributs
        self.data = None                # Données historiques
        self.X_train = None             # Données d'entraînement
        self.X_test = None              # Données de test
        self.y_train = None             # Valeurs cibles d'entraînement
        self.y_test = None              # Valeurs cibles de test
        
        # Ajout des attributs pour les données originales
        self.X_train_original = None    # Données d'entraînement originales
        self.X_test_original = None     # Données de test originales
        self.y_train_original = None    # Valeurs cibles d'entraînement originales
        self.y_test_original = None     # Valeurs cibles de test originales
        
        # Scalers séparés pour X et y
        self.X_scaler = StandardScaler()    # Scaler pour les données d'entrée
        self.y_scaler = StandardScaler()    # Scaler pour les valeurs cibles
        
        self.model = None                   # Modèle GPR initialisé à None pour l'instant 
        self.output_dir = "output"          # Répertoire de sortie pour les graphiques 
        os.makedirs(self.output_dir, exist_ok=True) # Crée le répertoire s'il n'existe pas déjà

    # Fonction pour télécharger les données historiques de l'actif
    def get_data(self):  
        """
        Récupère les données historiques pour l'actif.
        """
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date) # Télécharge les données historiques 
        print(f"Données téléchargées pour {self.ticker}: {len(self.data)} jours")      # Affiche le nombre de jours de données 
        return self.data                                                               # Retourne les données téléchargées

    # Fonction pour préparer les données avec une fenêtre glissante de 5 jours ( represantant les 5 jours de la semaine ou la bourse est ouverte)
    def prepare_data(self):
        """
        Prépare les données avec une fenêtre glissante.
        """
        X = []                  # Liste pour stocker les données d'entrée
        y = []                  # Liste pour stocker les valeurs cibles         
        for i in range(len(self.data) - self.window_size):                    # Parcours des données avec une fenêtre glissante
            X.append(self.data['Close'].iloc[i:i+self.window_size].values)    # Ajoute les données d'entrée
            y.append(self.data['Close'].iloc[i+self.window_size])             # Ajoute les valeurs cibles
            
        X = np.array(X)       # Convertit les données d'entrée en tableau numpy  
        y = np.array(y)       # Convertit les valeurs cibles en tableau numpy
        
        print("Shape of X:", X.shape)   # Affiche la forme des données d'entrée 
        print("Shape of y:", y.shape)   # Affiche la forme des valeurs cibles

        # Affichage de quelques exemples
        print("First 5 windows in X:\n", X[:5])
        print("First 5 target values in y:\n", y[:5])
        
        return X, y  # Retourne les données d'entrée et les valeurs cibles

    # Fonction pour diviser les données en ensembles d'entraînement et de test
    def train_test_split(self, X, y):
        """
        Divise les données en respectant l'ordre temporel.
        """
        split_idx = int(len(X) * (1 - self.test_size))              # Indice de division pour l'ensemble de test  
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]    # Division des données d'entrée  en ensembles d'entraînement et de test
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]    # Division des valeurs cibles  en ensembles d'entraînement et de test
        print(f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}") # Affiche la taille des ensembles d'entraînement et de test


    # Fonction pour normaliser les données tout en préservant leur structure
    def scale_data(self):
        """
        Normalise les données tout en préservant leur structure et garde une copie des données originales.
        """
        # Sauvegarde des données originales
        self.X_train_original = self.X_train.copy()   # Copie des données d'entraînement
        self.X_test_original = self.X_test.copy()     # Copie des données de test
        self.y_train_original = self.y_train.copy()   # Copie des valeurs cibles d'entraînement  
        self.y_test_original = self.y_test.copy()     # Copie des valeurs cibles de test   
        
        # Normalisation des données d'entrée (X)
        X_train_reshaped = self.X_train.reshape(-1, self.window_size)  # Remodelage des données d'entraînement 
        X_test_reshaped = self.X_test.reshape(-1, self.window_size)    # Remodelage des données de test
                
        X_train_scaled = self.X_scaler.fit_transform(X_train_reshaped)  # Normalisation des données d'entraînement 
        X_test_scaled = self.X_scaler.transform(X_test_reshaped)        # Normalisation des données de test 
        
        self.X_train = X_train_scaled  # Mise à jour des données d'entraînement
        self.X_test = X_test_scaled    # Mise à jour des données de test

        # Normalisation des valeurs cibles (y)
        self.y_train = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()  # Normalisation des valeurs cibles d'entraînement
        self.y_test = self.y_scaler.transform(self.y_test.reshape(-1, 1)).ravel()        # Normalisation des valeurs cibles de test   

    # Fonction pour effectuer les prédictions avec incertitudes et gérer la dénormalisation
    def predict(self): 
        """
        Effectue les prédictions avec incertitudes et gère la dénormalisation.
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions") # Vérifie si le modèle est entraîné avant de faire des prédictions 
        
        # Prédictions sur données normalisées
        predictions_norm, uncertainties_norm = self.model.predict(self.X_test, return_std=True)
        
        # Dénormalisation des prédictions
        predictions_denorm = self.y_scaler.inverse_transform(predictions_norm.reshape(-1, 1)).ravel() 
        
        # Ajustement des incertitudes
        uncertainties_denorm = uncertainties_norm * self.y_scaler.scale_[0]
        
        
        return predictions_denorm, uncertainties_denorm, self.y_test_original, self.y_train_original 
 
    # Fonction pour sauvegarder les résultats dans un graphique
    def save_plot(self, kernel_type, predictions, uncertainties):
        """
        Sauvegarde les résultats dans un graphique.
        """
        filename = f"{self.output_dir}/{self.ticker}_{kernel_type}_plot.png"  # Nom du fichier pour le graphique 
        fig = plot_predictions_with_uncertainty(                              # Crée le graphique avec les prédictions et les incertitudes  
            y_test=self.y_test_original,  # Utilisation des données originales 
            y_pred=predictions, 
            y_std=uncertainties,                    
            y_train=self.y_train_original  # Utilisation des données originales
        )
        fig.savefig(filename, bbox_inches='tight', dpi=300)  # Sauvegarde le graphique avec une résolution de 300 dpi
        print(f"Graph saved to {filename}") # Affiche le chemin du fichier 
        plt.close(fig)                      # Ferme la figure pour libérer la mémoire


    # Fonction pour  entrainer et evaluer le modèle avec différents noyaux 
    def evaluate_kernels(self):
        """
        Entraîne et évalue le modèle avec différents noyaux et sauvegarde les métriques en une seule fois.
        """
        # Affichage des statistiques générales avant de tester les noyaux
        print("\nStatistiques avant normalisation:")
        print(f"X_train - moyenne: {np.mean(self.X_train_original):.2f}, écart-type: {np.std(self.X_train_original):.2f}")
        print(f"y_train - moyenne: {np.mean(self.y_train_original):.2f}, écart-type: {np.std(self.y_train_original):.2f}")
        
        print("\nStatistiques après normalisation:")
        print(f"X_train - moyenne: {np.mean(self.X_train):.2f}, écart-type: {np.std(self.X_train):.2f}")
        print(f"y_train - moyenne: {np.mean(self.y_train):.2f}, écart-type: {np.std(self.y_train):.2f}")
        results = []  # Liste pour stocker les résultats des métriques

        # Dictionnaire des noyaux à tester
        kernels = {
            "RBF": get_rbf_kernel(),
            "Matern": get_matern_kernel(),
            "Rational Quadratic": get_rational_quadratic_kernel(),
            "Combined": get_combined_kernel()
        }

        for kernel_name, kernel in kernels.items():  # Parcours des noyaux à tester
            print(f"\nEntraînement avec le noyau : {kernel_name}")  # Affiche le noyau actuel
            
            # Réinitialisation du modèle avec le noyau actuel
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, random_state=0, optimizer=custom_optimizer) # Initialisation du modèle GPR avec le noyau actuel
            
            # Entraînement sur données normalisées
            self.model.fit(self.X_train, self.y_train) 
            
            # Prédictions et dénormalisation
            predictions_denorm, uncertainties_denorm, y_test_denorm, _ = self.predict()
            print("\nStatistiques après dénormalisation:")
            print(f"Predictions - moyenne: {np.mean(predictions_denorm):.2f}, écart-type: {np.std(predictions_denorm):.2f}") # Affiche les statistiques après dénormalisation 
            print(f"Original y_test - moyenne: {np.mean(self.y_test_original):.2f}, écart-type: {np.std(self.y_test_original):.2f}") # Affiche les statistiques après dénormalisation
            

            # Calcul des métriques sur données dénormalisées
            mse, rmse, mae, mape = calculate_metrics(y_test_denorm, predictions_denorm)
            
            # Récupération de la valeur de la vraisemblance marginale
            lml = self.model.log_marginal_likelihood_value_
            print(f"Log-Marginal Likelihood: {lml:.2f}")

            found_params = self.model.kernel_.get_params()  # Récupère les hyperparamètres trouvés

            # Sauvegarde du graphique avec un nom unique pour chaque noyau
            self.save_plot(kernel_name.lower(), predictions_denorm, uncertainties_denorm)

            # Stockage des résultats dans un dictionnaire
            results.append({
                "Kernel name": kernel_name,
                "RMSE": rmse,
                "MAE": mae,
                "MSE": mse,
                "MAPE": mape,
                "LML": lml,
                "Hyperparameters": found_params  # Enregistre les hyperparamètres trouvés
            })

        # Sauvegarde des résultats dans un DataFrame une fois après la boucle
        results_df = pd.DataFrame(results)
        results_filename = f"{self.output_dir}/kernel_metrics_results.csv"
        results_df.to_csv(results_filename, index=False)
        print(f"\nLes résultats des métriques ont été sauvegardés dans {results_filename}")

    
def main():
    financial_ml = FinancialML(
        ticker='IBM',  # IBM ( International Business Machines Corporation)
        start_date='2019-09-01',  # Date de début
        end_date='2022-09-01',    # Date de fin    
        window_size=5,            # Taille de la fenêtre   
        test_size=0.2             # Taille de l'ensemble de test  
    )
    
    financial_ml.get_data()        # Télécharge les données historiques  
    X, y = financial_ml.prepare_data() # Prépare les données avec une fenêtre glissante 
    financial_ml.train_test_split(X, y) # Divise les données en ensembles d'entraînement et de test
    financial_ml.scale_data()           # Normalise les données tout en préservant leur structure
    financial_ml.evaluate_kernels()     # Entraîne et évalue le modèle avec différents noyaux    

if __name__ == '__main__':
    main()