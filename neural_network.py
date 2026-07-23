import math
import random

def sigmoide(x):
    if x < -700:
        return 0.0
    if x > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def derivee_sigmoide(a):
    return a * (1.0 - a)

class MLP_Henon:
    def __init__(self, n_entree=2, n_cache=4, n_sortie=1, seed=None):
        self.n_entree = n_entree
        self.n_cache = n_cache
        self.n_sortie = n_sortie
        
        if seed is not None:
            random.seed(seed)
            
        # Initialisation de Xavier adaptée (Uniforme entre -limite et +limite)
        limite_1 = math.sqrt(6.0 / (n_entree + n_cache))
        limite_2 = math.sqrt(6.0 / (n_cache + n_sortie))
        
        # Poids W1 (entrée -> cachée) : matrice de taille [n_cache][n_entree]
        self.W1 = []
        for i in range(n_cache):
            ligne = []
            for j in range(n_entree):
                ligne.append(random.uniform(-limite_1, limite_1))
            self.W1.append(ligne)
            
        # Biais b1 (couche cachée) : vecteur de taille [n_cache]
        self.b1 = [0.0] * n_cache  # Initialisation à 0
        
        # Poids W2 (cachée -> sortie) : matrice de taille [n_sortie][n_cache]
        # Comme on a 1 seule sortie, c'est équivalent à un tableau de taille [n_cache]
        self.W2 = []
        for i in range(n_sortie):
            ligne = []
            for j in range(n_cache):
                ligne.append(random.uniform(-limite_2, limite_2))
            self.W2.append(ligne)
            
        # Biais b2 (couche de sortie) : vecteur de taille [n_sortie]
        self.b2 = [0.0] * n_sortie

        # Pour stocker les valeurs lors du forward pass (utiles pour le backward pass)
        self.entrees = [0.0] * n_entree
        self.z_cache = [0.0] * n_cache
        self.a_cache = [0.0] * n_cache
        self.z_sortie = [0.0] * n_sortie
        self.a_sortie = [0.0] * n_sortie
        
        # Pour le momentum (stockage des mises à jour précédentes)
        self.delta_W1 = [[0.0] * n_entree for _ in range(n_cache)]
        self.delta_b1 = [0.0] * n_cache
        self.delta_W2 = [[0.0] * n_cache for _ in range(n_sortie)]
        self.delta_b2 = [0.0] * n_sortie

    def forward_pass(self, entrees):
        # Sauvegarde des entrées pour le calcul du gradient
        for j in range(self.n_entree):
            self.entrees[j] = entrees[j]
            
        # --- 1. De la couche d'entrée vers la couche cachée ---
        for i in range(self.n_cache):
            somme = self.b1[i]
            for j in range(self.n_entree):
                somme += self.W1[i][j] * self.entrees[j]
            self.z_cache[i] = somme
            self.a_cache[i] = sigmoide(somme)  # Activation Sigmoïde
            
        # --- 2. De la couche cachée vers la couche de sortie ---
        for k in range(self.n_sortie):
            somme = self.b2[k]
            for i in range(self.n_cache):
                somme += self.W2[k][i] * self.a_cache[i]
            self.z_sortie[k] = somme
            self.a_sortie[k] = somme  # Activation Identité (f(z) = z)
            
        return list(self.a_sortie)  # Retourne une copie

    def backward_pass(self, cibles, lr, alpha=0.9):
        
        # --- 1. Calcul de l'erreur en sortie ---
        # error = sortie_prédite - cible_réelle
        # Comme l'activation est identité, f'(z_out) = 1.
        # Donc delta_sortie = error * 1
        delta_sortie = [0.0] * self.n_sortie
        for k in range(self.n_sortie):
            delta_sortie[k] = self.a_sortie[k] - cibles[k]
            
        # --- 2. Rétropropagation de l'erreur vers la couche cachée ---
        delta_cache = [0.0] * self.n_cache
        for i in range(self.n_cache):
            somme_erreurs = 0.0
            for k in range(self.n_sortie):
                somme_erreurs += self.W2[k][i] * delta_sortie[k]
            # Multiplication par la dérivée de la sigmoïde
            delta_cache[i] = somme_erreurs * derivee_sigmoide(self.a_cache[i])
            
        # --- 3. Mise à jour des poids (W2, b2) [Cachée -> Sortie] ---
        for k in range(self.n_sortie):
            # Mise à jour du biais b2
            grad_b2 = delta_sortie[k]
            step_b2 = lr * grad_b2 + alpha * self.delta_b2[k]
            self.b2[k] -= step_b2
            self.delta_b2[k] = step_b2
            
            # Mise à jour des poids W2
            for i in range(self.n_cache):
                grad_W2 = delta_sortie[k] * self.a_cache[i]
                step_W2 = lr * grad_W2 + alpha * self.delta_W2[k][i]
                self.W2[k][i] -= step_W2
                self.delta_W2[k][i] = step_W2
                
        # --- 4. Mise à jour des poids (W1, b1) [Entrée -> Cachée] ---
        for i in range(self.n_cache):
            # Mise à jour du biais b1
            grad_b1 = delta_cache[i]
            step_b1 = lr * grad_b1 + alpha * self.delta_b1[i]
            self.b1[i] -= step_b1
            self.delta_b1[i] = step_b1
            
            # Mise à jour des poids W1
            for j in range(self.n_entree):
                grad_W1 = delta_cache[i] * self.entrees[j]
                step_W1 = lr * grad_W1 + alpha * self.delta_W1[i][j]
                self.W1[i][j] -= step_W1
                self.delta_W1[i][j] = step_W1

def normaliser_serie(serie):
    s_min = serie[0]
    s_max = serie[0]
    for val in serie:
        if val < s_min:
            s_min = val
        if val > s_max:
            s_max = val
            
    amplitude = s_max - s_min
    if amplitude == 0:
        return [0.5 for _ in serie], s_min, s_max
        
    serie_norm = []
    for val in serie:
        serie_norm.append((val - s_min) / amplitude)
        
    return serie_norm, s_min, s_max

def denormaliser_valeur(val_norm, s_min, s_max):
    return val_norm * (s_max - s_min) + s_min

def creer_patterns_henon(serie_x):
    patterns_entree = []
    patterns_cible = []
    
    for n in range(2, len(serie_x)):
        entree = [serie_x[n-1], serie_x[n-2]]
        cible = serie_x[n]
        
        patterns_entree.append(entree)
        patterns_cible.append(cible)
        
    return patterns_entree, patterns_cible

if __name__ == "__main__":
    print("Test du module reseau de neurones...")
    reseau = MLP_Henon(n_entree=2, n_cache=4, n_sortie=1, seed=42)
    
    entrees_test = [0.5, 0.2]
    cible_test = [0.8]
    
    print("Avant entrainement :")
    sortie_avant = reseau.forward_pass(entrees_test)
    print("  Sortie :", sortie_avant)
    print("  Erreur :", sortie_avant[0] - cible_test[0])
    
    # Un pas d'entrainement
    reseau.backward_pass(cible_test, lr=0.1)
    
    print("Apres 1 pas d'entrainement :")
    sortie_apres = reseau.forward_pass(entrees_test)
    print("  Sortie :", sortie_apres)
    print("  Erreur :", sortie_apres[0] - cible_test[0])
    
    print("\nLe module neural_network.py est fonctionnel.")
