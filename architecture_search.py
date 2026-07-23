from henon_generator import generer_serie_henon
from neural_network import MLP_Henon, normaliser_serie, denormaliser_valeur, creer_patterns_henon
VALEURS_H = [4, 5, 6, 7, 8]
VALEURS_LR = [0.01, 0.03, 0.05, 0.07, 0.1]
N_RUNS = 3            # Nombre d'initialisations par configuration
MAX_EPOCHS = 3000
PATIENCE = 500        # Early stopping patience

# Division des données
N_TRAIN = 350
N_VAL = 148

def calculer_mse(reseau, patterns_entree, patterns_cible, s_min, s_max):
    mse = 0.0
    n = len(patterns_entree)
    if n == 0:
        return float('inf')
        
    for i in range(n):
        sortie_norm = reseau.forward_pass(patterns_entree[i])[0]
        # Dénormalisation
        sortie_reelle = denormaliser_valeur(sortie_norm, s_min, s_max)
        cible_reelle = denormaliser_valeur(patterns_cible[i], s_min, s_max)
        
        erreur = sortie_reelle - cible_reelle
        mse += erreur * erreur
        
    return mse / n

def entrainer_modele(H, lr, seed, X_train, Y_train, X_val, Y_val, s_min, s_max):
    reseau = MLP_Henon(n_entree=2, n_cache=H, n_sortie=1, seed=seed)
    
    meilleure_mse_val = float('inf')
    epoque_meilleure = 0
    epoque = 0
    
    while epoque < MAX_EPOCHS:
        # --- 1. Entraînement sur tout le dataset ---
        for i in range(len(X_train)):
            entree = X_train[i]
            cible = [Y_train[i]]
            reseau.forward_pass(entree)
            reseau.backward_pass(cible, lr, alpha=0.9) # Momentum
            
        # --- 2. Évaluation sur la validation (toutes les 10 époques) ---
        if epoque % 10 == 0:
            mse_val = calculer_mse(reseau, X_val, Y_val, s_min, s_max)
            
            if mse_val < meilleure_mse_val:
                meilleure_mse_val = mse_val
                epoque_meilleure = epoque
            elif epoque - epoque_meilleure >= PATIENCE:
                # Early Stopping
                break
                
        epoque += 1
        
    return meilleure_mse_val

if __name__ == "__main__":
    print("=" * 70)
    print("  RECHERCHE D'ARCHITECTURE OPTIMALE (GRID SEARCH)")
    print("=" * 70)
    
    print("\n[1] Preparation des donnees...")
    x_vals, _ = generer_serie_henon(1.4, 0.3, 0.0, 0.0, 500)
    
    x_norm, s_min, s_max = normaliser_serie(x_vals)
    print("  Plage de normalisation : [{:.4g}, {:.4g}]".format(s_min, s_max))
    
    X, Y = creer_patterns_henon(x_norm)  # 498 prototypes
    
    # Division chronologique (Train / Validation)
    X_train = X[:N_TRAIN]
    Y_train = Y[:N_TRAIN]
    X_val = X[N_TRAIN:N_TRAIN+N_VAL]
    Y_val = Y[N_TRAIN:N_TRAIN+N_VAL]
    
    print("  Entrainement : {} patterns".format(len(X_train)))
    print("  Validation   : {} patterns".format(len(X_val)))
    
    # --- Grid Search ---
    print("\n[2] Lancement du Grid Search ({} configurations * {} runs)...".format(
        len(VALEURS_H) * len(VALEURS_LR), N_RUNS))
    
    resultats = []
    
    # Affichage en-tête tableau
    print("\n" + "-" * 65)
    print("  H |   lr   |  MSE Val (Run 1, 2, 3)    |  MSE Val (Moyenne)")
    print("-" * 65)
    
    meilleure_config = None
    meilleure_mse_globale = float('inf')
    
    # Stockage pour le tableau récapitulatif
    tableau_recap = {}
    
    for H in VALEURS_H:
        tableau_recap[H] = {}
        for lr in VALEURS_LR:
            mses_runs = []
            
            for run in range(N_RUNS):
                seed = 42 + run * 100 + H * 10 + int(lr * 100) # Graine unique
                mse = entrainer_modele(H, lr, seed, X_train, Y_train, X_val, Y_val, s_min, s_max)
                mses_runs.append(mse)
                
            mse_moyenne = sum(mses_runs) / N_RUNS
            tableau_recap[H][lr] = mse_moyenne
            
            # Affichage ligne
            print(" {:2d} |  {:.2f}  | {:.8g}, {:.8g}, {:.8g} |  {:.8g}".format(
                H, lr, mses_runs[0], mses_runs[1], mses_runs[2], mse_moyenne))
                
            if mse_moyenne < meilleure_mse_globale:
                meilleure_mse_globale = mse_moyenne
                meilleure_config = (H, lr)
                
    print("-" * 65)
    
    print("\n[3] TABLEAU RECAPITULATIF : MSE VALIDATION MOYENNE")
    print("-" * 75)
    ligne_entete = "  H / lr  |"
    for lr in VALEURS_LR:
        ligne_entete += "    {:.2f}    |".format(lr)
    print(ligne_entete)
    print("-" * 75)
    
    for H in VALEURS_H:
        ligne = "    {:2d}    |".format(H)
        for lr in VALEURS_LR:
            val = tableau_recap[H][lr]
            if (H, lr) == meilleure_config:
                ligne += " *{:.6g}* |".format(val)
            else:
                ligne += "  {:.6g}  |".format(val)
        print(ligne)
    print("-" * 75)
    
    print("\n[CONCLUSION] Architecture optimale trouvee :")
    print("  -> Nombre de neurones caches (H) : {}".format(meilleure_config[0]))
    print("  -> Taux d'apprentissage (lr)     : {}".format(meilleure_config[1]))
    print("  -> MSE Validation correspondante : {:.8g}".format(meilleure_mse_globale))
    
    with open("best_architecture.txt", "w") as f:
        f.write("{},{}\n".format(meilleure_config[0], meilleure_config[1]))
        f.write("{:.8g}\n".format(meilleure_mse_globale))
