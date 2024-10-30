import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def plot_predictions_with_uncertainty(y_test, y_pred, y_std, y_train):
    """
    Trace les prédictions avec les intervalles d'incertitude.
    """
    plt.figure(figsize=(15, 8))
    
    # Créer les indices temporels
    train_indices = np.arange(len(y_train))
    test_indices = np.arange(len(y_train), len(y_train) + len(y_test))
    
    # Configuration de la grille
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_axisbelow(True)  # Mettre la grille en arrière-plan
    
    # Tracer les différentes séries
    plt.plot(train_indices, y_train, label='Training Series', color='lightblue', linewidth=1)
    plt.plot(test_indices, y_test, label='Testing Series', color='orange', linewidth=1)
    plt.plot(test_indices, y_pred, label='Predictions', color='blue', linewidth=2)
    
    # Ajouter l'intervalle de confiance
    plt.fill_between(test_indices,
                    y_pred - 2*y_std,
                    y_pred + 2*y_std,
                    color='blue', alpha=0.15,
                    label='Confidence Interval')
    
    # Personnalisation du graphique
    plt.title('Time Series Prediction with Uncertainty', fontsize=14, pad=15)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    
    # Ajuster la légende
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99),
              framealpha=0.9, fontsize=10)
    
    # Calculer et afficher les métriques
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    """# Afficher les métriques dans une boîte semi-transparente
    metrics_text = f'MSE: {mse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}'
    plt.text(0.01, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             fontsize=10,
             verticalalignment='top')"""
     
    # Ajuster les limites et les marges
    plt.margins(x=0.01)
    
    # Ajuster la mise en page
    plt.tight_layout()
    plt.show()

def plot_training_progress(training_losses, val_losses=None):
    """
    Trace l'évolution de la perte pendant l'entraînement.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()