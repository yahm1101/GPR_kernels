# monthly_portfolio.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from visualisation import plot_predictions_with_uncertainty
from kernels import get_rbf_kernel, get_matern_kernel, get_rational_quadratic_kernel, get_combined_kernel
from train import custom_optimizer
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

 # Classe pour le portefeuille mensuel 
class MonthlyPortfolioML:

    def __init__(self, tickers: list, start_date: str, end_date: str, window_size: int = 5):
        self.tickers = tickers            # Liste des actifs
        self.start_date = start_date      
        self.end_date = end_date
        self.window_size = window_size
        
        # Dictionnaires pour stocker les données et modèles par actif
        self.data = {}                    # Données mensuelles par actif (actif,prix de clôture,rendements)
        self.models = {}                  # Modèles GPR par actif/kernel ( actif,kernel) je stock les modeles GPR entrainés pour chaque actif et chaque kernel
        self.scalers = {}                 # Scalers par actif ( actif,scaler) je stock les scalers utilisés pour normaliser les données d'entraînement et de test
        
        # Pour les résultats
        self.predictions = {} # Prédictions ( actif,kernel) je stock les prédictions de rendements pour chaque actif et chaque kernel
        self.uncertainties = {} # Incertitudes ( actif,kernel) je stock les incertitudes(ecarts-types) associées aux prédictions de rendements pour chaque actif et chaque kernel
        self.returns = {}  # Rendements par( actif) je stock les rendements mensuels réels pour chaque actif

    def normalize_data(self):
        """
        Normalise les données d'entraînement et de test combinées, y compris les cibles.
        """
        self.global_scaler = StandardScaler()
        self.X_train = self.global_scaler.fit_transform(self.X_train)
        self.X_test = self.global_scaler.transform(self.X_test)

          # Normalisation des cibles (y)
        self.y_scaler = StandardScaler()
        self.y_train = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_test = self.y_scaler.transform(self.y_test.reshape(-1, 1)).flatten()

    def get_monthly_data(self):  
        """
        Télécharge les données mensuelles pour tous les actifs.
        """
        for ticker in self.tickers:  # Pour chaque actif dans la liste des actifs 
            # Données journalières
            daily_data = yf.download(ticker, start=self.start_date, end=self.end_date) # Téléchargement des données journalières 
            # Conversion en données mensuelles (dernier jour du mois)
            monthly_data = daily_data.resample('M').last()   # Conversion en données mensuelles  ( on prend le dernier jour du mois)
            # Calcul des rendements mensuels
            # Ra,tj d´enote le retour sur investissant dans l’actif a entre t − 1 et t 

            monthly_data['Returns'] = (monthly_data['Close'] - monthly_data['Close'].shift(1)) / monthly_data['Close'].shift(1)  # Calcul des rendements mensuels (Ra,t)
            self.data[ticker] = monthly_data # Stockage des données dans le dictionnaire 
            print(f"Données téléchargées pour {ticker}: {len(monthly_data)} mois")  # Affichage du nombre de mois téléchargés 

    def prepare_sequences(self, ticker):
        """
        Prépare les séquences xa,t et ya,t pour un actif donné.
        """
        data = self.data[ticker]['Returns'].dropna()  # Rendements mensuels (Ra,t)
        X = []  # séquences de rendements (xa,t)
        y = []  # rendements cibles (ya,t)

        for i in range(self.window_size, len(data)):
            x_sequence = data.iloc[i-self.window_size:i].values  # Equation (6)
            
            y_target = data.iloc[i]  # le rendement a predire
            X.append(x_sequence)  # Ajout de la séquence xa,t
            y.append(y_target)   # Ajout du rendement cible ya,t

        return np.array(X), np.array(y)  # Retourne les séquences xa,t et ya,t non normalisées
    
    def prepare_all_data(self): # Combine les séquences pour tous les actifs et prépare les ensembles d'entraînement et de test
        """
        Prépare les données selon les équations (13) et (14)
        """
        X_all_train = []  # Séquences d'entraînement pour tous les actifs
        y_all_train = []  # Rendements cibles d'entraînement pour tous les actifs
        X_all_test = []     # Séquences de test pour tous les actifs
        y_all_test = []     # Rendements cibles de test pour tous les actifs 

        for ticker in self.tickers:
            # Préparation des séquences pour cet actif
            X, y = self.prepare_sequences(ticker)  # Préparation des séquences xa,t et ya,t pour l'actif a
            
            # Division train/test
            train_size = int(len(X) * 0.8) # 80% des données pour l'entraînement et 20% pour le test
            
            # Ajout aux ensembles d'entraînement et de test globaux
            X_all_train.extend(X[:train_size])  # Ajout des séquences d'entraînement pour l'actif a
            y_all_train.extend(y[:train_size])  # Ajout des rendements cibles d'entraînement pour l'actif a 
            X_all_test.extend(X[train_size:])   # Ajout des séquences de test pour l'actif a
            y_all_test.extend(y[train_size:])  # Ajout des rendements cibles de test pour l'actif a

        # Conversion en arrays numpy
        self.X_train = np.array(X_all_train)  # tableau numpy pour les séquences d'entraînement
        self.y_train = np.array(y_all_train)  # tableau numpy pour les rendements cibles d'entraînement
        self.X_test = np.array(X_all_test)   # tableau numpy pour les séquences de test
        self.y_test = np.array(y_all_test)   # tableau numpy pour les rendements cibles de test

        # Normaliser globalement après la combinaison
        self.normalize_data()



    def inverse_transform_targets(self, predictions):
        """
        Transforms the normalized predictions back to their original scale.
        This should only be applied to the predictions, not during training or testing.
        """
        if hasattr(self, 'y_scaler') and self.y_scaler:
            # Use the y_scaler to transform normalized predictions back to the original scale
            return self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        else:
            raise ValueError("y_scaler is not defined. Ensure normalize_data() is called before this.")

    
    def train_models(self):
        """ 
        Entraîne chaque kernel sur l'ensemble Dr complet
        """
        self.models = {}  # Un modèle par kernel (plus de séparation par actif)

        # Définition des kernels à tester
        kernels = {   # Les kernels mesurent non seulement la relation temporelle d’un actif, mais aussi les similarités à travers différents actifs.
            "RBF": get_rbf_kernel(),
            "Matern": get_matern_kernel(),
            "Rational Quadratic": get_rational_quadratic_kernel(),
           # "Combined": get_combined_kernel()
        }

        # Pour chaque kernel
        for kernel_name, kernel in kernels.items():
            print(f"\nEntraînement avec {kernel_name} sur tous les actifs")
            
            # Création du modèle GPR
            model = GaussianProcessRegressor(   # la matrice de covariance utilisée par le modèle GPR capture la similarité entre les séquences des actifs. 
                kernel=kernel,
                n_restarts_optimizer=40,
                random_state=0,
                optimizer=custom_optimizer
            )

            # Entraînement sur l'ensemble Dr complet
            model.fit(self.X_train, self.y_train)

            self.models[kernel_name] = model
            print(f"Log-Marginal Likelihood: {model.log_marginal_likelihood_value_:.2f}")  # Affichage de la vraisemblance marginale 

    def predict_returns(self):
        """
        Prédictions sur l'ensemble Ds complet pour chaque kernel.
        """
        self.predictions = {}
        self.uncertainties = {}

        # Pour chaque kernel
        for kernel_name, model in self.models.items():
            # Prédictions sur tout l'ensemble de test
            pred_norm, std_norm = model.predict(self.X_test, return_std=True)
            
            # Dénormalisation des prédictions
            pred_denorm = self.y_scaler.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
            std_denorm = std_norm * self.y_scaler.scale_

        
            # Stockage des prédictions et incertitudes pour ce kernel
            self.predictions[kernel_name] = pred_denorm
            self.uncertainties[kernel_name] = std_denorm

    def generate_kernel_tables(self):
        """
        Crée les tableaux selon l'équation (23) du PDF avec les poids calculés selon (20) et (21).
        """
        self.kernel_tables = {}
        n_test_periods = len(self.X_test) // len(self.tickers)
        
        # Inverse transform y_test once
        y_test_denorm = self.inverse_transform_targets(self.y_test)
        
        for kernel_name, model in self.models.items():
            kernel_table = pd.DataFrame(index=range(n_test_periods))
            
            for t in range(n_test_periods):
                scores_t = {}
                for i, ticker in enumerate(self.tickers):
                    index = i * n_test_periods + t
                    μ = self.predictions[kernel_name][index]
                    σ = self.uncertainties[kernel_name][index]
                    scores_t[ticker] = max(0, μ / σ) if σ != 0 else 0
                
                total_score = sum(scores_t.values())
                if total_score != 0:
                    weights_t = {ticker: score / total_score for ticker, score in scores_t.items()}
                else:
                    weights_t = {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
                
                for i, ticker in enumerate(self.tickers):
                    index = i * n_test_periods + t
                    Ra_t = y_test_denorm[index]
                    kernel_table.loc[t, f'{ticker}(w*R)'] = weights_t[ticker] * Ra_t
            
            kernel_table['R(κ)π,t'] = kernel_table.sum(axis=1)
            self.kernel_tables[kernel_name] = kernel_table


    


    def calculate_total_returns(self):  # Calcul des rendements totaux (eq.25) 
        """
        Équation (25) du PDF : calcule (1 + R(κ)π,Ts) = ∏(1 + R(κ)π,t)
        """
        self.total_returns = {} # dictionnaire pour stocker les rendements totaux par kernel (kernel,rendement total)
        
        for kernel_name, table in self.kernel_tables.items():  # Pour chaque kernel
            portfolio_returns = table['R(κ)π,t']  # Récupération des rendements totaux pour ce kernel
            total_return = np.prod(1 + portfolio_returns) - 1  # Calcul du rendement total (eq. 25)
            self.total_returns[kernel_name] = total_return  # Stockage du rendement total pour ce kernel

    def calculate_sharpe_ratios(self):  # Calcul des ratios de Sharpe (eq.26)
        """
        Équation (26) du PDF : calcule S(κ)π,Ts = R̄(κ)π / σ[R(κ)π,t]
        """
        self.sharpe_ratios = {}  # Dictionnaire pour stocker les ratios de Sharpe par kernel
        
        for kernel_name, table in self.kernel_tables.items():  # Pour chaque kernel
            portfolio_returns = table['R(κ)π,t']  # Série des rendements
            
            mean_return = np.mean(portfolio_returns)  # Moyenne des rendements
            std_dev = np.std(portfolio_returns)  # Écart-type des rendements
            
            if std_dev != 0:
                sharpe = mean_return / std_dev  # Calcul du ratio de Sharpe
            else:
                sharpe = 0  # Ou gérer le cas où l'écart-type est zéro
            
            self.sharpe_ratios[kernel_name] = sharpe  # Stockage du ratio de Sharpe pour ce kernel


    def display_results(self):
        """
        Affiche les tableaux et les métriques finales.
        """
        # Affiche d'abord les tableaux détaillés
        print("\nTableaux détaillés par kernel :")
        for kernel_name, table in self.kernel_tables.items():
            print(f"\n{kernel_name} :")
            print(table.round(4))  # Affichage des tableaux avec 4 décimales
        
        # Puis les métriques finales
        results = []
        for kernel_name in self.models.keys():
            results.append({  # Stockage des résultats dans une liste de dictionnaires pour créer un DataFrame
                'Kernel': kernel_name,
                'Total Return ': f"{self.total_returns[kernel_name]:.2%}", #affichage en pourcentage du rendement total pour chaque kernel
                'Sharpe Ratio ': f"{self.sharpe_ratios[kernel_name]:.4f}"    #affichage du ratio de Sharpe pour chaque kernel 
            })
        
        results_df = pd.DataFrame(results)  # Création d'un DataFrame à partir de la liste de dictionnaires
        print("\nRésultats finaux :")
        print(results_df.to_string(index=False)) # Affichage des résultats finaux sous forme de tableau sans index 

    def run_analysis(self):
        """
        Séquence d'exécution mise à jour.
        """
        print("Téléchargement des données...")
        self.get_monthly_data()
        
        print("\nPréparation des données...")
        self.prepare_all_data()
        
        print("\nEntraînement des modèles...")
        self.train_models()
        
        print("\nCalcul des prédictions...")
        self.predict_returns()
        
        print("\nGénération des tableaux par kernel...")
        self.generate_kernel_tables()
        
        print("\nCalcul des rendements totaux (eq.25)...")
        self.calculate_total_returns()
        
        print("\nCalcul des ratios de Sharpe (eq.26)...")
        self.calculate_sharpe_ratios()
        
        print("\nAffichage des résultats...")
        self.display_results()

if __name__ == '__main__':
        # Liste des actifs à analyser
        tickers = ['RBLX', 'ZM', 'PINS', 'SNAP', 'PLTR']
        
        portfolio = MonthlyPortfolioML(
            tickers=tickers,
            start_date='2016-01-01',
            end_date='2023-12-31',
            window_size=5  # 5 mois de fenêtre glissante
        )
        
        portfolio.run_analysis()                     