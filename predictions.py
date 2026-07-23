import os
import math
from henon_generator import generer_serie_henon
from neural_network import MLP_Henon, denormaliser_valeur, sigmoide

def charger_poids(reseau, chemin_fichier="poids_finaux.txt"):
    if not os.path.exists(chemin_fichier):
        print(f"[ERREUR] Le fichier {chemin_fichier} n'existe pas. Entrainez le modele d'abord.")
        return False
        
    try:
        with open(chemin_fichier, 'r') as f:
            lignes = f.readlines()
            
        donnees = [l.strip() for l in lignes if not l.startswith("#") and l.strip()]
        
        # Structure du fichier :
        # H lignes pour W1
        # 1 ligne pour b1
        # 1 ligne pour W2
        # 1 ligne pour b2
        
        idx = 0
        for i in range(reseau.n_cache):
            valeurs = [float(x) for x in donnees[idx].split(',')]
            for j in range(reseau.n_entree):
                reseau.W1[i][j] = valeurs[j]
            idx += 1
            
        valeurs_b1 = [float(x) for x in donnees[idx].split(',')]
        for i in range(reseau.n_cache):
            reseau.b1[i] = valeurs_b1[i]
        idx += 1
        
        for k in range(reseau.n_sortie):
            valeurs_W2 = [float(x) for x in donnees[idx].split(',')]
            for i in range(reseau.n_cache):
                reseau.W2[k][i] = valeurs_W2[i]
            idx += 1
            
        valeurs_b2 = [float(x) for x in donnees[idx].split(',')]
        for k in range(reseau.n_sortie):
            reseau.b2[k] = valeurs_b2[k]
            
        print("[OK] Poids charges avec succes.")
        return True
    except Exception as e:
        print(f"[ERREUR] Impossible de charger les poids : {e}")
        return False

def predire_1_pas(reseau, X_test, Y_test, s_min, s_max):
    predictions = []
    mse = 0.0
    mae = 0.0
    n = len(X_test)
    
    for i in range(n):
        # Forward pass (les entrées sont déjà normalisées)
        sortie_norm = reseau.forward_pass(X_test[i])[0]
        
        # Dénormalisation
        pred_reelle = denormaliser_valeur(sortie_norm, s_min, s_max)
        cible_reelle = denormaliser_valeur(Y_test[i], s_min, s_max)
        
        erreur = pred_reelle - cible_reelle
        mse += erreur * erreur
        mae += abs(erreur)
        
        predictions.append(pred_reelle)
        
    return predictions, mse / n, mae / n

def predire_k_pas(reseau, historiques_norm, s_min, s_max, k_pas):
    fenetre = list(historiques_norm) # Copie
    
    for step in range(k_pas):
        entree = [fenetre[-1], fenetre[-2]]
        sortie_norm = reseau.forward_pass(entree)[0]
        
        fenetre.append(sortie_norm)
        
    pred_finale_norm = fenetre[-1]
    
    return denormaliser_valeur(pred_finale_norm, s_min, s_max)

def evaluer_multi_pas(reseau, serie_norm, s_min, s_max, k_pas, idx_debut, idx_fin):
    predictions = []
    cibles = []
    mse = 0.0
    mae = 0.0
    
    compteur = 0
    for n in range(idx_debut, idx_fin - k_pas + 1):
        historique = [serie_norm[n-2], serie_norm[n-1]] 
        
        cible_norm = serie_norm[n - 1 + k_pas]
        cible_reelle = denormaliser_valeur(cible_norm, s_min, s_max)
        
        pred_reelle = predire_k_pas(reseau, historique, s_min, s_max, k_pas)
        
        erreur = pred_reelle - cible_reelle
        mse += erreur * erreur
        mae += abs(erreur)
        
        predictions.append(pred_reelle)
        cibles.append(cible_reelle)
        compteur += 1
        
    return predictions, cibles, mse / compteur, mae / compteur

def afficher_tableau_predictions(cibles, predictions, mse, mae, n_afficher=10, horizon=1):
    print("\n" + "=" * 80)
    print("  PREDICTION A {} PAS EN AVANT".format(horizon))
    print("=" * 80)
    ligne_sep = "+" + "-" * 10 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 20 + "+"
    print(ligne_sep)
    print("|{:^10s}|{:^20s}|{:^20s}|{:^20s}|".format("Point", "Valeur Reelle", "Valeur Predite", "Erreur Absolue"))
    print(ligne_sep)
    
    for i in range(min(n_afficher, len(cibles))):
        err = abs(predictions[i] - cibles[i])
        print("| {:>8d} | {:>18.8g} | {:>18.8g} | {:>18.8g} |".format(
            i+1, cibles[i], predictions[i], err))
            
    print(ligne_sep)
    print("  METRIQUES GLOBALES :")
    print("  -> Erreur Quadratique Moyenne (MSE) : {:.8g}".format(mse))
    print("  -> Racine de MSE (RMSE)             : {:.8g}".format(math.sqrt(mse)))
    print("  -> Erreur Absolue Moyenne (MAE)     : {:.8g}".format(mae))
    print("=" * 80)

if __name__ == "__main__":
    print("=" * 65)
    print("  EVALUATION DES PREDICTIONS (SERIE DE HENON)")
    print("=" * 65)
    
    x_vals, _ = generer_serie_henon(1.4, 0.3, 0.0, 0.0, 500)
    
    try:
        with open("norm_params.txt", "r") as f:
            ligne = f.readline()
            s_min = float(ligne.split(',')[0])
            s_max = float(ligne.split(',')[1])
    except:
        print("[!] norm_params.txt non trouve, calcul a la volee.")
        from neural_network import normaliser_serie
        _, s_min, s_max = normaliser_serie(x_vals)
        
    x_norm = []
    amplitude = s_max - s_min
    for val in x_vals:
        x_norm.append((val - s_min) / amplitude)
        
    H_OPTIMAL = 6
    if os.path.exists("best_architecture.txt"):
        with open("best_architecture.txt", "r") as f:
            lignes = f.readlines()
            if len(lignes) >= 1:
                H_OPTIMAL = int(lignes[0].split(',')[0].strip())
                
    reseau = MLP_Henon(n_entree=2, n_cache=H_OPTIMAL, n_sortie=1)
    
    if charger_poids(reseau, "poids_finaux.txt"):
        
        N_TRAIN = 350
        N_VAL = 148
        IDX_DEBUT_TEST = 352
        IDX_FIN_TEST = 500
        
        # ---  PREDICTIONS 1 PAS ---
        from neural_network import creer_patterns_henon
        X_all, Y_all = creer_patterns_henon(x_norm)
        X_test = X_all[N_TRAIN:N_TRAIN+N_VAL]
        Y_test = Y_all[N_TRAIN:N_TRAIN+N_VAL]
        
        pred_1, mse_1, mae_1 = predire_1_pas(reseau, X_test, Y_test, s_min, s_max)
        
        cibles_1 = [denormaliser_valeur(y, s_min, s_max) for y in Y_test]
        
        afficher_tableau_predictions(cibles_1, pred_1, mse_1, mae_1, horizon=1)
        
        # ---  PREDICTIONS 3 PAS ---
        pred_3, cibles_3, mse_3, mae_3 = evaluer_multi_pas(
            reseau, x_norm, s_min, s_max, k_pas=3, idx_debut=IDX_DEBUT_TEST, idx_fin=IDX_FIN_TEST)
        afficher_tableau_predictions(cibles_3, pred_3, mse_3, mae_3, horizon=3)
        
        # ---  PREDICTIONS 10 PAS ---
        pred_10, cibles_10, mse_10, mae_10 = evaluer_multi_pas(
            reseau, x_norm, s_min, s_max, k_pas=10, idx_debut=IDX_DEBUT_TEST, idx_fin=IDX_FIN_TEST)
        afficher_tableau_predictions(cibles_10, pred_10, mse_10, mae_10, horizon=10)
        
        # ---  PREDICTIONS 20 PAS ---
        pred_20, cibles_20, mse_20, mae_20 = evaluer_multi_pas(
            reseau, x_norm, s_min, s_max, k_pas=20, idx_debut=IDX_DEBUT_TEST, idx_fin=IDX_FIN_TEST)
        afficher_tableau_predictions(cibles_20, pred_20, mse_20, mae_20, horizon=20)
        
        print("\n" + "=" * 80)
        print("  ANALYSE ET REMARQUES SUR LES RESULTATS")
        print("=" * 80)
        print("Evolution de l'erreur MSE :")
        print(" - 1 pas  : {:.8g}".format(mse_1))
        print(" - 3 pas  : {:.8g} (Facteur x{:.2f})".format(mse_3, mse_3/mse_1 if mse_1 > 0 else 0))
        print(" - 10 pas : {:.8g} (Facteur x{:.2f})".format(mse_10, mse_10/mse_1 if mse_1 > 0 else 0))
        print(" - 20 pas : {:.8g} (Facteur x{:.2f})".format(mse_20, mse_20/mse_1 if mse_1 > 0 else 0))
        print("\nConclusion :")
        print("La serie de Henon est un systeme dynamique chaotique. Le fait que l'erreur")
        print("augmente de facon exponentielle avec l'horizon de prediction n'est pas un defaut")
        print("du reseau de neurones, mais une propriete intrinseque du chaos connue sous le")
        print("nom de 'Sensibilite aux conditions initiales'. Toute petite erreur a l'etape n")
        print("(due a l'approximation de la fonction) est amplifiee lors des iterations futures.")
        print("Au bout de 20 pas, on depasse l'horizon de predictibilite (Temps de Lyapunov),")
        print("et la prediction deterministe devient caduque.")
        print("=" * 80)
        
    else:
        print("Veuillez d'abord executer training.py.")
