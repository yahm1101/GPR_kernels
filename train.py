import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from visualisation import plot_predictions_with_uncertainty
from kernels import get_rbf_kernel, get_matern_kernel, get_rational_quadratic_kernel, get_combined_kernel

 
class FinancialML: # Classe qui va permettre de faire des prédictions sur les données financières
    def __init__(self, ticker: str, start_date: str, end_date: str, window_size: int = 5, test_size: float = 0.2): # Initialisation de la classe avec les paramètres de base 
        """
        Initialise la classe avec les paramètres de base.
        """
        self.ticker = ticker # Nom de l'actif
        self.start_date = start_date # Date de début
        self.end_date = end_date # Date de fin
        self.window_size = window_size # Taille de la fenêtre
        self.test_size = test_size # Taille du jeu de test
        
        # Initialisation des attributs qui seront remplis plus tard
        self.data = None # Données historiques
        self.X_train = None # Données d'entraînement
        self.X_test = None # Données de test
        self.y_train = None # Valeurs cibles d'entraînement  
        self.y_test = None # Valeurs cibles de test
        self.scaler = StandardScaler() # Normalisation des données
        self.model = None # Modèle de régression
        
    def get_data(self): 
        """
        Récupère les données historiques pour l'actif.
        """
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date) # Téléchargement des données
        print(f"Données téléchargées pour {self.ticker}: {len(self.data)} jours") # Affichage du nombre de jours
        return self.data 
    
    def prepare_data(self):
        """
        Prépare les données avec une fenêtre glissante.
        """
        X = [] 
        y = [] 
        for i in range(len(self.data) - self.window_size): # Parcours des données 
            X.append(self.data['Close'].iloc[i:i+self.window_size].values) # Ajout des valeurs de la fenêtre
            y.append(self.data['Close'].iloc[i+self.window_size]) # Ajout de la valeur cible
        return np.array(X), np.array(y) # Retourne les données sous forme de tableau
    
    def train_test_split(self, X, y): # Division des données en données d'entraînement et de test
        """
        Divise les données en respectant l'ordre temporel.
        """
        split_idx = int(len(X) * (1 - self.test_size)) # Calcul de l'indice de séparation
        self.X_train, self.X_test = X[:split_idx], X[split_idx:] # Division des données
        self.y_train, self.y_test = y[:split_idx], y[split_idx:] # Division des valeurs cibles
        print(f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}") # Affichage des tailles
        
    def scale_data(self):
        """
        Normalise les données tout en préservant la structure des fenêtres.
        """
        # Reshape pour appliquer StandardScaler sur l'ensemble des données d'entrée
        X_train_reshaped = self.X_train.reshape(-1, self.window_size)
        X_test_reshaped = self.X_test.reshape(-1, self.window_size)
        
        # Normalisation des données d'entrée (X)
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        # Remettre à la forme originale
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled

        # Initialisation d'un scaler séparé pour y_train et y_test
        self.y_scaler = StandardScaler()
        self.y_train = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_test = self.y_scaler.transform(self.y_test.reshape(-1, 1)).flatten()



    def train_model(self, kernel_type='rbf'): # Entraînement du modèle avec le kernel spécifié
        """
        Entraîne le modèle GPR avec le kernel spécifié.
        """
        # Sélection du kernel
        if kernel_type == 'rbf':
            kernel = get_rbf_kernel()
        elif kernel_type == 'matern':
            kernel = get_matern_kernel()
        elif kernel_type == 'rational_quadratic':
            kernel = get_rational_quadratic_kernel()
        elif kernel_type == 'combined':
            kernel = get_combined_kernel()
        else:
            raise ValueError("Type de kernel non supporté")
        
        # Création et entraînement du modèle
        self.model = GaussianProcessRegressor(kernel=kernel, random_state=0)
        self.model.fit(self.X_train, self.y_train)
        print("Modèle entraîné avec succès")
    
    def predict(self):
        """
        Effectue les prédictions avec incertitudes.
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        predictions, uncertainties = self.model.predict(self.X_test, return_std=True)
        
        # Dénormaliser les prédictions
        predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
        uncertainties = uncertainties * self.y_scaler.scale_[0]  # Ajustement des incertitudes pour la cible
        
        return predictions, uncertainties

    
    def run_pipeline(self, kernel_type='rational_quadratic'):
        """
        Exécute tout le pipeline d'apprentissage.
        """
        # 1. Récupération des données
        self.get_data()
        
        # 2. Préparation des données
        X, y = self.prepare_data()
        
        # 3. Division train/test
        self.train_test_split(X, y)
        
        # 4. Normalisation
        self.scale_data()
        
        # 5. Entraînement du modèle
        self.train_model(kernel_type)
        
        # 6. Prédictions
        predictions, uncertainties = self.predict()
        
       # 7. Visualisation avec données d'entraînement
        plot_predictions_with_uncertainty(
            y_test=self.y_test,
            y_pred=predictions,
            y_std=uncertainties,
            y_train=self.y_train  # Ajout des données d'entraînement
        )
        return predictions, uncertainties

def main():
    # Création de l'instance
    financial_ml = FinancialML(
        ticker='META',
        start_date='2010-01-01',
        end_date='2023-09-01',
        window_size=5,
        test_size=0.2
    )
    
    # Exécution du pipeline complet
    predictions, uncertainties = financial_ml.run_pipeline()

if __name__ == '__main__':
    main()