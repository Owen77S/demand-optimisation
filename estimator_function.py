"""
Fonction estimateur de temps de calcul pour appli.calculation()
Generee automatiquement par benchmark_analysis.py

Modele: Regression lineaire avec normalisation StandardScaler

Metriques du modele:
- R2 (Test): 0.9620
- RMSE (Test): 74.7597 secondes
- MAE (Test): 25.8549 secondes
"""

import numpy as np

# Parametres de normalisation (StandardScaler)
SCALER_MEAN = np.array([np.float64(2.0714285714285716), np.float64(1.0476190476190477), np.float64(45.714285714285715), np.float64(45.714285714285715), np.float64(45.714285714285715), np.float64(281309.5238095238), np.float64(684000.0)])
SCALER_SCALE = np.array([np.float64(0.8561502188745335), np.float64(0.5753831415997415), np.float64(40.599252699130936), np.float64(40.599252699130936), np.float64(40.599252699130936), np.float64(640586.9885576017), np.float64(1912778.2088340707)])

# Coefficients du modele
COEFFICIENTS = np.array([np.float64(3.0679393385016156), np.float64(0.9508212035349821), np.float64(-0.24379924968143596), np.float64(0.34698624026198127), np.float64(-0.151336058829474), np.float64(40.891686702807554), np.float64(-7.638738250890927)])
INTERCEPT = 20.943485


def estimate_calculation_time(nb_configs, nb_fixeds, len_pop, n_iter, n_sample):
    """
    Estime le temps de calcul de la fonction appli.calculation()

    Parameters
    ----------
    nb_configs : int
        Nombre de configurations
    nb_fixeds : int
        Nombre de fixeds
    len_pop : int
        Taille de la population
    n_iter : int
        Nombre d'iterations
    n_sample : int
        Nombre d'echantillons

    Returns
    -------
    float
        Temps estime en secondes
    """
    # Creer le vecteur de features brutes
    product = nb_configs * len_pop * n_iter * n_sample
    product_fixeds = nb_configs * (nb_fixeds + 1) * len_pop * n_iter * n_sample

    features_raw = np.array([
        nb_configs,
        nb_fixeds,
        len_pop,
        n_iter,
        n_sample,
        product,
        product_fixeds
    ])

    # Normaliser les features
    features_normalized = (features_raw - SCALER_MEAN) / SCALER_SCALE

    # Prediction
    estimated_time = INTERCEPT + np.dot(COEFFICIENTS, features_normalized)

    return estimated_time


if __name__ == "__main__":
    # Exemple d'utilisation
    nb_configs = 2
    nb_fixeds = 1
    len_pop = 50
    n_iter = 50
    n_sample = 50

    estimated_time = estimate_calculation_time(nb_configs, nb_fixeds, len_pop, n_iter, n_sample)
    print(f"Temps estime: {estimated_time:.2f} secondes")
