A = 1.4          
B = 0.3         
X0 = 0.0         
Y0 = 0.0         
N_POINTS = 500   

def generer_serie_henon(a, b, x0, y0, n):
    liste_x = [0.0] * n
    liste_y = [0.0] * n

    liste_x[0] = x0
    liste_y[0] = y0

    # Génération par récurrence
    for i in range(1, n):
        liste_x[i] = liste_y[i - 1] + 1.0 - a * liste_x[i - 1] * liste_x[i - 1]
        liste_y[i] = b * liste_x[i - 1]

    return liste_x, liste_y


def exporter_csv(liste_x, liste_y, chemin_fichier):
    fichier = open(chemin_fichier, 'w')
    fichier.write("n,x_n,y_n\n")
    for i in range(len(liste_x)):
        ligne = str(i) + "," + "{:.8g}".format(liste_x[i]) + "," + "{:.8g}".format(liste_y[i]) + "\n"
        fichier.write(ligne)
    fichier.close()
    print("[OK] Donnees exportees dans : " + chemin_fichier)


def afficher_tableau(liste_x, liste_y, debut=0, fin=None):
    if fin is None:
        fin = len(liste_x)

    ligne_sep = "+" + "-" * 7 + "+" + "-" * 20 + "+" + "-" * 20 + "+"
    print(ligne_sep)
    print("|{:^7s}|{:^20s}|{:^20s}|".format("n", "x_n", "y_n"))
    print(ligne_sep)

    for i in range(debut, fin):
        print("|{:^7d}| {:>18.8g} | {:>18.8g} |".format(i, liste_x[i], liste_y[i]))

    print(ligne_sep)


def afficher_statistiques(liste_x, liste_y):
    n = len(liste_x)

    x_min = liste_x[0]
    x_max = liste_x[0]
    x_somme = 0.0
    for i in range(n):
        if liste_x[i] < x_min:
            x_min = liste_x[i]
        if liste_x[i] > x_max:
            x_max = liste_x[i]
        x_somme += liste_x[i]

    y_min = liste_y[0]
    y_max = liste_y[0]
    y_somme = 0.0
    for i in range(n):
        if liste_y[i] < y_min:
            y_min = liste_y[i]
        if liste_y[i] > y_max:
            y_max = liste_y[i]
        y_somme += liste_y[i]

    x_moy = x_somme / n
    y_moy = y_somme / n

    print("")
    print("=" * 65)
    print("  STATISTIQUES DE LA SERIE DE HENON ({} points)".format(n))
    print("=" * 65)
    print("  x_n : min = {:>12.8g}  |  max = {:>12.8g}  |  moy = {:>12.8g}".format(
        x_min, x_max, x_moy))
    print("  y_n : min = {:>12.8g}  |  max = {:>12.8g}  |  moy = {:>12.8g}".format(
        y_min, y_max, y_moy))
    print("  Amplitude x : {:.8g}".format(x_max - x_min))
    print("  Amplitude y : {:.8g}".format(y_max - y_min))
    print("=" * 65)


if __name__ == "__main__":
    print("=" * 65)
    print("  GENERATION DE LA SERIE DE HENON")
    print("  a = {}, b = {}, x_0 = {}, y_0 = {}".format(A, B, X0, Y0))
    print("  Nombre de points : {}".format(N_POINTS))
    print("=" * 65)

    x_vals, y_vals = generer_serie_henon(A, B, X0, Y0, N_POINTS)

    print("\n--- 20 premieres valeurs ---")
    afficher_tableau(x_vals, y_vals, 0, 20)

    print("\n--- 20 dernieres valeurs ---")
    afficher_tableau(x_vals, y_vals, 480, 500)

    afficher_statistiques(x_vals, y_vals)

    exporter_csv(x_vals, y_vals, "henon_500.csv")

    print("\n[OK] Generation terminee avec succes.")
