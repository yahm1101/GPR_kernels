import matplotlib.pyplot as plt
import numpy as np


def plot_predictions_with_uncertainty(y_test, y_pred, y_std, y_train): # y_train added to the function signature to plot training data as well
    """
    Trace les prédictions avec les intervalles d'incertitude.
    """
    plt.figure(figsize=(20,10)) # Ajuster la taille du graphique
    
    # Créer les indices temporels
    train_indices = np.arange(len(y_train)) # Créer un tableau d'indices pour les données d'entraînement 
    test_indices = np.arange(len(y_train), len(y_train) + len(y_test)) # Créer un tableau d'indices pour les données de test
    
    # Configuration de la grille
    plt.grid(True, linestyle='--', alpha=0.7) # Ajouter une grille en pointillés
    plt.gca().set_axisbelow(True)  # Mettre la grille en arrière-plan pour ne pas cacher les données
    
    # Tracer les différentes séries
    plt.plot(train_indices, y_train, label='Training Series', color='lightblue', linewidth=1) # Tracer les données d'entraînement
    plt.plot(test_indices, y_test, label='Testing Series', color='orange', linewidth=1) # Tracer les données de test
    plt.plot(test_indices, y_pred, label='Predictions', color='blue', linewidth=2) # Tracer les prédictions
    
    # Ajouter l'intervalle de confiance
    plt.fill_between(test_indices, # Remplir l'intervalle entre les bornes inférieures et supérieures
                    y_pred - 2*y_std, # Limite inférieure
                    y_pred + 2*y_std, # Limite supérieure
                    color='blue', alpha=0.15, # Couleur et transparence
                    label='Confidence Interval') # Légende
    
    # Personnalisation du graphique (étiquette des axes, titre, légende)
    plt.title('Time Series Prediction with Uncertainty', fontsize=14, pad=15) 
    plt.xlabel('Time Step', fontsize=12) 
    plt.ylabel('Value', fontsize=12)
    
    # Ajuster la légende 
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99),
              framealpha=0.9, fontsize=10) 
    
    # Ajuster les limites et les marges
    plt.margins(x=0.01) # Ajuster les marges horizontales
    
    # Ajuster la mise en page
    plt.tight_layout()  # Ajuster la mise en page pour éviter les coupures
    
    return plt.gcf() # Retourner la figure pour une personnalisation ultérieure

    
def plot_training_progress(training_losses, val_losses=None):
    """
    Trace l'évolution de la perte pendant l'entraînement.
    """
    plt.figure(figsize=(10, 5)) # Ajuster la taille du graphique
    plt.plot(training_losses, label='Training Loss') # Tracer la perte d'entraînement
    if val_losses is not None: # Check if validation losses are provided
        plt.plot(val_losses, label='Validation Loss') # Plot validation losses if available
    plt.xlabel('Epoch') # Étiquette de l'axe des x
    plt.ylabel('Loss') # Étiquette de l'axe des y
    plt.title('Training Progress') # Titre du graphique
    plt.legend() 
    plt.grid(True) # Ajouter une grille
    plt.show()