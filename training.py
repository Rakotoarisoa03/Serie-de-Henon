import os
from henon_generator import generer_serie_henon
from neural_network import MLP_Henon, normaliser_serie, creer_patterns_henon, denormaliser_valeur
import architecture_search # pour utiliser la fonction calculer_mse

MAX_EPOCHS = 10000    
PATIENCE = 1000       
SEED = 42             

# Si le fichier n'existe pas, on met des valeurs par défaut (ex: H=6, lr=0.05)
H_OPTIMAL = 6
LR_OPTIMAL = 0.05

if os.path.exists("best_architecture.txt"):
    with open("best_architecture.txt", "r") as f:
        lignes = f.readlines()
        if len(lignes) >= 2:
            H_OPTIMAL = int(lignes[0].split(',')[0].strip())
            LR_OPTIMAL = float(lignes[0].split(',')[1].strip())

def sauvegarder_poids(reseau, chemin_fichier="poids_finaux.txt"):
    with open(chemin_fichier, 'w') as f:
        f.write("# Poids Entree -> Cachee (W1)\n")
        for i in range(reseau.n_cache):
            ligne = ",".join(["{:.8g}".format(w) for w in reseau.W1[i]])
            f.write(ligne + "\n")
            
        f.write("# Biais Cachee (b1)\n")
        ligne = ",".join(["{:.8g}".format(b) for b in reseau.b1])
        f.write(ligne + "\n")
        
        f.write("# Poids Cachee -> Sortie (W2)\n")
        for k in range(reseau.n_sortie):
            ligne = ",".join(["{:.8g}".format(w) for w in reseau.W2[k]])
            f.write(ligne + "\n")
            
        f.write("# Biais Sortie (b2)\n")
        ligne = ",".join(["{:.8g}".format(b) for b in reseau.b2])
        f.write(ligne + "\n")
        
    print("[OK] Poids sauvegardes dans : " + chemin_fichier)

if __name__ == "__main__":
    print("=" * 65)
    print("  APPRENTISSAGE FINAL DU RESEAU DE NEURONES")
    print("=" * 65)
    print("Architecture choisie :")
    print("  -> Neurones caches (H) :", H_OPTIMAL)
    print("  -> Taux d'apprentissage :", LR_OPTIMAL)
    print("  -> Max Epochs :", MAX_EPOCHS)
    print("-" * 65)
    
    x_vals, _ = generer_serie_henon(1.4, 0.3, 0.0, 0.0, 500)
    x_norm, s_min, s_max = normaliser_serie(x_vals)
    X, Y = creer_patterns_henon(x_norm)
    
    N_TRAIN = 350
    N_VAL = 148
    X_train, Y_train = X[:N_TRAIN], Y[:N_TRAIN]
    X_val, Y_val = X[N_TRAIN:N_TRAIN+N_VAL], Y[N_TRAIN:N_TRAIN+N_VAL]
    
    with open("norm_params.txt", "w") as f:
        f.write("{:.8g},{:.8g}\n".format(s_min, s_max))
    
    reseau = MLP_Henon(n_entree=2, n_cache=H_OPTIMAL, n_sortie=1, seed=SEED)
    
    print("\nLancement de l'entrainement...")
    meilleure_mse_val = float('inf')
    epoque_meilleure = 0
    
    historique_train = []
    historique_val = []
    
    for epoque in range(MAX_EPOCHS):
        # Apprentissage
        for i in range(len(X_train)):
            reseau.forward_pass(X_train[i])
            reseau.backward_pass([Y_train[i]], LR_OPTIMAL)
            
        # Évaluation (toutes les 50 époques pour affichage)
        if epoque % 50 == 0 or epoque == MAX_EPOCHS - 1:
            mse_train = architecture_search.calculer_mse(reseau, X_train, Y_train, s_min, s_max)
            mse_val = architecture_search.calculer_mse(reseau, X_val, Y_val, s_min, s_max)
            
            historique_train.append(mse_train)
            historique_val.append(mse_val)
            
            if mse_val < meilleure_mse_val:
                meilleure_mse_val = mse_val
                epoque_meilleure = epoque
                # On sauvegarde les meilleurs poids en mémoire (simplifié: on sauvegarde le modèle à la fin)
                
            print("  Epoque {:5d}/{} | MSE Train: {:.8g} | MSE Val: {:.8g}".format(
                epoque, MAX_EPOCHS, mse_train, mse_val))
                
            # Early Stopping
            if epoque - epoque_meilleure >= PATIENCE:
                print("\n[!] Early Stopping declenche a l'epoque {} (meilleure val a {})".format(epoque, epoque_meilleure))
                break
                
    print("\n" + "=" * 65)
    print("  BILAN DE L'ENTRAINEMENT")
    print("=" * 65)
    print("  -> Meilleure epoque :", epoque_meilleure)
    print("  -> MSE Validation finale : {:.8g}".format(meilleure_mse_val))
    
    sauvegarder_poids(reseau)
    print("\n[OK] Entrainement final termine avec succes.")
