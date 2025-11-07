"""
Script pour estimer le temps de calcul de la fonction calculation (appli.py)
en fonction de ses paramètres principaux via une régression linéaire.

Paramètres analysés :
- len(configs) : Nombre de configurations (1-4)
- nb_fixeds : Nombre de fixeds (0-2)
- len_pop : Taille de la population (10-100, avec échantillon à 200)
- n_iter : Nombre d'itérations (10-100, avec échantillon à 200)
- n_sample : Nombre d'échantillons (10-100, avec échantillon à 200)
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import test_package as test
import appli

# Configuration
FILEPATH = 'data/power_fixed.csv'
COLUMN = 2
TYPE_PROFILE = 'monthly'
DEVELOPMENT = False

# Grille de paramètres pour le benchmark d'entraînement
# Format: (nb_configs, nb_fixeds, len_pop, n_iter, n_sample)
def generate_training_grid():
    """Génère une grille variée de paramètres pour l'entraînement."""
    training_grid = []

    # Tests rapides (baseline - faibles valeurs)
    training_grid.extend([
        (1, 0, 10, 10, 10),
        (1, 1, 10, 10, 10),
        (1, 2, 10, 10, 10),
        (2, 0, 10, 10, 10),
        (2, 1, 10, 10, 10),
    ])

    # Variation de nb_configs avec paramètres moyens
    training_grid.extend([
        (1, 1, 30, 30, 30),
        (2, 1, 30, 30, 30),
        (3, 1, 30, 30, 30),
        (4, 1, 30, 30, 30),
    ])

    # Variation de nb_fixeds
    training_grid.extend([
        (2, 0, 30, 30, 30),
        (2, 2, 30, 30, 30),
        (3, 0, 30, 30, 30),
        (3, 2, 30, 30, 30),
    ])

    # Variation de len_pop
    training_grid.extend([
        (2, 1, 20, 30, 30),
        (2, 1, 40, 30, 30),
        (2, 1, 60, 30, 30),
        (2, 1, 80, 30, 30),
        (2, 1, 100, 30, 30),
    ])

    # Variation de n_iter
    training_grid.extend([
        (2, 1, 30, 20, 30),
        (2, 1, 30, 40, 30),
        (2, 1, 30, 60, 30),
        (2, 1, 30, 80, 30),
        (2, 1, 30, 100, 30),
    ])

    # Variation de n_sample
    training_grid.extend([
        (2, 1, 30, 30, 20),
        (2, 1, 30, 30, 40),
        (2, 1, 30, 30, 60),
        (2, 1, 30, 30, 80),
        (2, 1, 30, 30, 100),
    ])

    # Combinaisons intermédiaires variées
    training_grid.extend([
        (1, 0, 50, 50, 50),
        (3, 1, 40, 40, 40),
        (4, 2, 30, 30, 30),
        (1, 2, 70, 40, 30),
        (3, 0, 30, 70, 40),
        (2, 2, 40, 30, 70),
    ])

    # Tests avec valeurs élevées (200)
    training_grid.extend([
        (1, 1, 200, 30, 30),
        (2, 1, 30, 200, 30),
        (2, 1, 30, 30, 200),
        (1, 0, 200, 50, 50),
        (1, 1, 50, 200, 50),
        (1, 1, 50, 50, 200),
    ])

    # Tests aux limites supérieures
    training_grid.extend([
        (4, 2, 100, 100, 100),
        (3, 2, 80, 80, 80),
        (4, 1, 60, 60, 60),
    ])

    return training_grid


def generate_test_grid():
    """Génère une grille distincte pour tester le modèle."""
    test_grid = [
        # Combinaisons non vues pendant l'entraînement
        (1, 1, 25, 25, 25),
        (2, 0, 35, 45, 55),
        (3, 2, 45, 35, 25),
        (4, 0, 15, 15, 15),
        (2, 2, 50, 50, 50),
        (3, 1, 70, 30, 40),
        (1, 2, 90, 60, 70),
        (4, 1, 40, 80, 60),
        (2, 1, 65, 65, 65),
        (3, 0, 55, 75, 45),
        # Cas extrêmes pour validation
        (1, 0, 200, 100, 100),
        (4, 2, 200, 200, 200),
    ]
    return test_grid


TRAINING_GRID = generate_training_grid()
TEST_GRID = generate_test_grid()


def run_benchmark(nb_configs, nb_fixeds, len_pop, n_iter, n_sample):
    """
    Exécute un benchmark avec les paramètres donnés et retourne le temps d'exécution.

    Parameters
    ----------
    nb_configs : int
        Nombre de configurations à créer (1-4)
    nb_fixeds : int
        Nombre de fixeds à créer (0-2)
    len_pop : int
        Taille de la population pour l'algorithme génétique
    n_iter : int
        Nombre d'itérations
    n_sample : int
        Nombre d'échantillons

    Returns
    -------
    float
        Temps d'exécution en secondes
    """
    # Créer des configurations et fixeds aléatoires
    configs = test.create_configs(nb_configs)
    fixeds = test.create_fixeds(nb_fixeds) if nb_fixeds > 0 else []

    # Paramètres de l'optimisation
    parameters = [len_pop, n_iter, n_sample]

    # Mesurer le temps
    start_time = time.time()
    try:
        power_profile, best_schedule = appli.calculation(
            FILEPATH, COLUMN, TYPE_PROFILE, configs, fixeds, DEVELOPMENT, parameters
        )
        elapsed_time = time.time() - start_time
        return elapsed_time
    except Exception as e:
        print(f"  [ERREUR] {e}")
        return None


def collect_data(parameter_grid, dataset_name="training"):
    """
    Collecte les données de temps pour tous les paramètres de la grille.

    Parameters
    ----------
    parameter_grid : list of tuples
        Liste de tuples (nb_configs, nb_fixeds, len_pop, n_iter, n_sample)
    dataset_name : str
        Nom du dataset ("training" ou "test")

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les paramètres et les temps mesurés
    """
    results = []
    total = len(parameter_grid)

    print(f"\nDémarrage du benchmark {dataset_name.upper()} avec {total} configurations...\n")

    for idx, (nb_configs, nb_fixeds, len_pop, n_iter, n_sample) in enumerate(parameter_grid, 1):
        print(f"[{idx}/{total}] nb_configs={nb_configs}, nb_fixeds={nb_fixeds}, "
              f"len_pop={len_pop}, n_iter={n_iter}, n_sample={n_sample}")

        elapsed_time = run_benchmark(nb_configs, nb_fixeds, len_pop, n_iter, n_sample)

        if elapsed_time is not None:
            results.append({
                'nb_configs': nb_configs,
                'nb_fixeds': nb_fixeds,
                'len_pop': len_pop,
                'n_iter': n_iter,
                'n_sample': n_sample,
                'time_seconds': elapsed_time
            })
            print(f"  [OK] Temps mesure: {elapsed_time:.2f}s\n")
        else:
            print(f"  [ERREUR] Echec du benchmark\n")

    df = pd.DataFrame(results)
    return df


def perform_regression(df_train, df_test=None):
    """
    Effectue une régression linéaire multiple sur les données.

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame d'entraînement
    df_test : pd.DataFrame, optional
        DataFrame de test séparé

    Returns
    -------
    model : LinearRegression
        Modèle de régression entraîné
    metrics : dict
        Métriques d'évaluation du modèle
    """
    # Préparer les features et la target pour l'entraînement
    feature_cols = ['nb_configs', 'nb_fixeds', 'len_pop', 'n_iter', 'n_sample']
    X_train = df_train[feature_cols].values
    y_train = df_train['time_seconds'].values

    # Ajouter des features dérivées (interactions)
    # Feature 1: produit de tous les paramètres (complexité théorique)
    product_train = (df_train['nb_configs'] *
                     df_train['len_pop'] *
                     df_train['n_iter'] *
                     df_train['n_sample']).values.reshape(-1, 1)

    # Feature 2: produit incluant nb_fixeds
    product_with_fixeds_train = (df_train['nb_configs'] *
                                  (df_train['nb_fixeds'] + 1) *
                                  df_train['len_pop'] *
                                  df_train['n_iter'] *
                                  df_train['n_sample']).values.reshape(-1, 1)

    # Créer une matrice de features étendue
    X_train_extended = np.hstack([
        X_train,  # Features originales
        product_train,  # Produit sans fixeds
        product_with_fixeds_train,  # Produit avec fixeds
    ])

    # Entraîner le modèle
    model = LinearRegression()
    model.fit(X_train_extended, y_train)

    # Prédictions sur l'entraînement
    y_pred_train = model.predict(X_train_extended)

    # Préparer les données de test si fournies
    if df_test is not None and len(df_test) > 0:
        X_test = df_test[feature_cols].values
        y_test = df_test['time_seconds'].values

        product_test = (df_test['nb_configs'] *
                       df_test['len_pop'] *
                       df_test['n_iter'] *
                       df_test['n_sample']).values.reshape(-1, 1)

        product_with_fixeds_test = (df_test['nb_configs'] *
                                     (df_test['nb_fixeds'] + 1) *
                                     df_test['len_pop'] *
                                     df_test['n_iter'] *
                                     df_test['n_sample']).values.reshape(-1, 1)

        X_test_extended = np.hstack([
            X_test,
            product_test,
            product_with_fixeds_test,
        ])

        y_pred_test = model.predict(X_test_extended)
    else:
        X_test_extended = None
        y_test = None
        y_pred_test = None

    # Calculer les métriques
    metrics = {
        'r2_train': r2_score(y_train, y_pred_train),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }

    if y_test is not None:
        metrics['r2_test'] = r2_score(y_test, y_pred_test)
        metrics['rmse_test'] = np.sqrt(mean_squared_error(y_test, y_pred_test))
        metrics['mae_test'] = mean_absolute_error(y_test, y_pred_test)

    return model, metrics, X_train_extended, y_train, y_pred_train, X_test_extended, y_test, y_pred_test


def visualize_results(df_train, df_test, model, metrics, X_train_ext, y_train, y_pred_train, X_test_ext, y_test, y_pred_test):
    """
    Crée des visualisations pour évaluer le modèle.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Analyse de la Régression Linéaire - Estimation du Temps de Calcul', fontsize=16, fontweight='bold')

    # 1. Prédictions vs Valeurs Réelles (Train + Test)
    ax1 = axes[0, 0]
    ax1.scatter(y_train, y_pred_train, alpha=0.6, label='Train', color='blue', s=50)
    if y_test is not None:
        ax1.scatter(y_test, y_pred_test, alpha=0.6, label='Test', color='red', s=50)

    # Ligne parfaite y=x
    all_vals = np.concatenate([y_train, y_pred_train])
    if y_test is not None:
        all_vals = np.concatenate([all_vals, y_test, y_pred_test])
    min_val = all_vals.min()
    max_val = all_vals.max()
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Prédiction parfaite')

    ax1.set_xlabel('Temps réel (s)', fontsize=11)
    ax1.set_ylabel('Temps prédit (s)', fontsize=11)
    ax1.set_title('Prédictions vs Valeurs Réelles', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Résidus
    ax2 = axes[0, 1]
    residuals_train = y_train - y_pred_train
    ax2.scatter(y_pred_train, residuals_train, alpha=0.6, label='Train', color='blue', s=50)

    if y_test is not None:
        residuals_test = y_test - y_pred_test
        ax2.scatter(y_pred_test, residuals_test, alpha=0.6, label='Test', color='red', s=50)

    ax2.axhline(y=0, color='k', linestyle='--', lw=2)
    ax2.set_xlabel('Temps prédit (s)', fontsize=11)
    ax2.set_ylabel('Résidus (s)', fontsize=11)
    ax2.set_title('Analyse des Résidus', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Importance des features (coefficients)
    ax3 = axes[0, 2]
    feature_names = ['nb_configs', 'nb_fixeds', 'len_pop', 'n_iter', 'n_sample', 'product', 'product_fixeds']
    coeffs = metrics['coefficients']
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    bars = ax3.barh(feature_names, coeffs, color=colors)
    ax3.set_xlabel('Coefficient', fontsize=11)
    ax3.set_title('Importance des Paramètres (Coefficients)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # Ajouter les valeurs sur les barres
    for bar in bars:
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.2e}', ha='left' if width >= 0 else 'right',
                va='center', fontsize=8)

    # 4. Distribution des temps mesurés (Train)
    ax4 = axes[1, 0]
    ax4.hist(df_train['time_seconds'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Temps de calcul (s)', fontsize=11)
    ax4.set_ylabel('Fréquence', fontsize=11)
    ax4.set_title('Distribution des Temps (Train)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Erreur relative (%)
    ax5 = axes[1, 1]
    rel_error_train = np.abs((y_train - y_pred_train) / y_train) * 100
    ax5.hist(rel_error_train, bins=15, alpha=0.7, color='lightgreen', edgecolor='black', label='Train')

    if y_test is not None:
        rel_error_test = np.abs((y_test - y_pred_test) / y_test) * 100
        ax5.hist(rel_error_test, bins=15, alpha=0.7, color='salmon', edgecolor='black', label='Test')

    ax5.set_xlabel('Erreur relative (%)', fontsize=11)
    ax5.set_ylabel('Fréquence', fontsize=11)
    ax5.set_title('Distribution des Erreurs Relatives', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Métriques comparatives
    ax6 = axes[1, 2]
    ax6.axis('off')

    metrics_text = "METRIQUES RECAPITULATIVES\n" + "="*40 + "\n\n"
    metrics_text += f"Train:\n"
    metrics_text += f"  R² = {metrics['r2_train']:.4f}\n"
    metrics_text += f"  RMSE = {metrics['rmse_train']:.2f} s\n"
    metrics_text += f"  MAE = {metrics['mae_train']:.2f} s\n\n"

    if 'r2_test' in metrics:
        metrics_text += f"Test:\n"
        metrics_text += f"  R² = {metrics['r2_test']:.4f}\n"
        metrics_text += f"  RMSE = {metrics['rmse_test']:.2f} s\n"
        metrics_text += f"  MAE = {metrics['mae_test']:.2f} s\n\n"

    metrics_text += f"Intercept = {metrics['intercept']:.4f}"

    ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('time_estimation_analysis.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Graphique sauvegarde: time_estimation_analysis.png")

    return fig


def print_results(metrics):
    """
    Affiche les résultats de la régression de manière formatée.
    """
    print("\n" + "="*80)
    print(" "*20 + "RESULTATS DE LA REGRESSION LINEAIRE")
    print("="*80 + "\n")

    print("METRIQUES DE PERFORMANCE:")
    print("-" * 80)
    print(f"  R² (Train): {metrics['r2_train']:.4f}")
    if 'r2_test' in metrics:
        print(f"  R² (Test):  {metrics['r2_test']:.4f}")

    print(f"\n  RMSE (Train): {metrics['rmse_train']:.4f} secondes")
    if 'rmse_test' in metrics:
        print(f"  RMSE (Test):  {metrics['rmse_test']:.4f} secondes")

    print(f"\n  MAE (Train): {metrics['mae_train']:.4f} secondes")
    if 'mae_test' in metrics:
        print(f"  MAE (Test):  {metrics['mae_test']:.4f} secondes")

    print("\nMODELE DE REGRESSION:")
    print("-" * 80)
    print(f"  Intercept: {metrics['intercept']:.6f}")
    print("\n  Coefficients:")
    feature_names = ['nb_configs', 'nb_fixeds', 'len_pop', 'n_iter', 'n_sample',
                     'product', 'product_with_fixeds']
    for name, coef in zip(feature_names, metrics['coefficients']):
        print(f"    {name:20s}: {coef:.6e}")

    print("\nFORMULE DE L'ESTIMATEUR:")
    print("-" * 80)
    c = metrics['coefficients']
    i = metrics['intercept']
    print(f"\n  T(nb_configs, nb_fixeds, len_pop, n_iter, n_sample) = ")
    print(f"      {i:.6f}")
    print(f"    + {c[0]:.6e} × nb_configs")
    print(f"    + {c[1]:.6e} × nb_fixeds")
    print(f"    + {c[2]:.6e} × len_pop")
    print(f"    + {c[3]:.6e} × n_iter")
    print(f"    + {c[4]:.6e} × n_sample")
    print(f"    + {c[5]:.6e} × (nb_configs × len_pop × n_iter × n_sample)")
    print(f"    + {c[6]:.6e} × (nb_configs × (nb_fixeds+1) × len_pop × n_iter × n_sample)")

    print("\n" + "="*80 + "\n")


def main():
    """
    Fonction principale pour exécuter le benchmark complet.
    """
    print("="*80)
    print(" "*15 + "BENCHMARK - ESTIMATION DU TEMPS DE CALCUL")
    print("="*80 + "\n")

    # Étape 1: Collecter les données d'entraînement
    print("ÉTAPE 1: Collection des données d'ENTRAÎNEMENT")
    print("-" * 80)
    df_train = collect_data(TRAINING_GRID, "training")
    df_train.to_csv('benchmark_data_train.csv', index=False)
    print(f"\n[OK] Donnees d'entrainement sauvegardees: benchmark_data_train.csv")
    print(f"[OK] Nombre d'observations (train): {len(df_train)}\n")

    # Étape 2: Collecter les données de test
    print("\nÉTAPE 2: Collection des données de TEST")
    print("-" * 80)
    df_test = collect_data(TEST_GRID, "test")
    df_test.to_csv('benchmark_data_test.csv', index=False)
    print(f"\n[OK] Donnees de test sauvegardees: benchmark_data_test.csv")
    print(f"[OK] Nombre d'observations (test): {len(df_test)}\n")

    # Étape 3: Régression linéaire
    print("\nÉTAPE 3: Régression linéaire")
    print("-" * 80)
    model, metrics, X_train_ext, y_train, y_pred_train, X_test_ext, y_test, y_pred_test = perform_regression(df_train, df_test)

    # Étape 4: Visualisation
    print("\nÉTAPE 4: Visualisation des résultats")
    print("-" * 80)
    fig = visualize_results(df_train, df_test, model, metrics, X_train_ext, y_train, y_pred_train, X_test_ext, y_test, y_pred_test)

    # Étape 5: Affichage des résultats
    print_results(metrics)

    # Interpretation
    print("INTERPRETATION:")
    print("-" * 80)
    r2_train = metrics['r2_train']
    r2_test = metrics.get('r2_test', r2_train)

    if r2_test > 0.95:
        print("  [EXCELLENT] Le modele explique tres bien les variations du temps.")
    elif r2_test > 0.85:
        print("  [BON] Le modele est fiable pour estimer le temps de calcul.")
    elif r2_test > 0.70:
        print("  [ACCEPTABLE] Le modele donne une estimation raisonnable.")
    else:
        print("  [INSUFFISANT] Le modele lineaire ne capture pas bien la complexite.")

    print("\n  Analyse des coefficients:")
    dominant_idx = np.argmax(np.abs(metrics['coefficients']))
    feature_names = ['nb_configs', 'nb_fixeds', 'len_pop', 'n_iter', 'n_sample',
                     'product', 'product_with_fixeds']
    print(f"    - Feature dominante: {feature_names[dominant_idx]}")

    if dominant_idx >= 5:
        print("    - Le terme produit domine -> Complexite multiplicative confirmee")

    print("\n  Recommandations:")
    print("    - Pour ameliorer la precision, augmenter la taille du dataset")
    print("    - Tester des modeles non-lineaires (polynomial, Ridge, Lasso, etc.)")
    if abs(r2_train - r2_test) > 0.1:
        print("    - Ecart train/test important -> Risque de surapprentissage")

    print("\n" + "="*80)
    print("[OK] Benchmark termine avec succes!")
    print("="*80 + "\n")

    return df_train, df_test, model, metrics


if __name__ == "__main__":
    df_train, df_test, model, metrics = main()
