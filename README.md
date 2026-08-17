# 🧠 Prédiction de la série de Hénon avec un réseau de neurones

Projet universitaire réalisé en **Python natif** pour modéliser et prédire la série temporelle chaotique de **Hénon** à l'aide d'un **réseau de neurones multicouches (MLP)**.

> L'objectif principal est d'implémenter le fonctionnement du réseau de neurones **sans utiliser de bibliothèque de Machine Learning** afin de mieux comprendre son fonctionnement interne.

## 🚀 Application

L'application Web permet de visualiser les données de Hénon, l'attracteur chaotique, l'architecture du réseau ainsi que les résultats des prédictions.

👉 **[Voir l'application en ligne](https://serie-de-henon.vercel.app/)**

## 🎯 Objectifs

* Générer les données de la série de Hénon.
* Construire un réseau de neurones MLP entièrement à la main.
* Rechercher automatiquement la meilleure architecture.
* Entraîner le réseau.
* Effectuer des prédictions à **1, 3, 10 et 20 pas**.
* Observer les limites de la prédiction d'un système chaotique.

## 🧠 Architecture

Le réseau obtenu après recherche d'architecture est :

```text
2 neurones → 5 neurones cachés → 1 neurone
```

* **Entrées :** `xₙ` et `xₙ₋₁`
* **Couche cachée :** 5 neurones avec fonction sigmoïde
* **Sortie :** prédiction de `xₙ₊₁`
* **Learning rate :** `0.03`
* **Momentum :** `0.9`

La meilleure architecture a été déterminée par un **Grid Search**.

## ⚙️ Technologies

* **Python**
* **HTML / CSS**
* **JavaScript**
* **Chart.js**

Le réseau est implémenté sans :

```text
NumPy
Pandas
TensorFlow
PyTorch
Scikit-learn
```

Les principaux mécanismes sont codés manuellement :

* Forward Pass
* Backpropagation
* Initialisation Xavier/Glorot
* Momentum
* Early Stopping
* Normalisation Min-Max

## 📁 Structure

```text
├── henon_generator.py
├── architecture_search.py
├── neural_network.py
├── training.py
├── predictions.py
│
├── best_architecture.txt
├── norm_params.txt
├── poids_finaux.txt
│
└── interface/
    ├── data/
    │   └── henon_500.csv
    └── index.html
```

### Rôle des scripts

| Fichier                  | Fonction                            |
| ------------------------ | ----------------------------------- |
| `henon_generator.py`     | Génère la série de Hénon            |
| `architecture_search.py` | Recherche la meilleure architecture |
| `neural_network.py`      | Contient le réseau MLP              |
| `training.py`            | Entraîne le réseau                  |
| `predictions.py`         | Effectue les prédictions            |
| `interface/index.html`   | Interface Web                       |

## 📊 Résultats

| Prédiction |           MSE |
| ---------- | ------------: |
| **1 pas**  | `6.42 × 10⁻⁶` |
| **3 pas**  | `2.51 × 10⁻⁴` |
| **10 pas** |       `0.223` |
| **20 pas** |       `1.001` |

Le réseau est **très précis à court terme**, mais l'erreur augmente fortement lorsque le nombre de pas augmente.

Cette divergence illustre la **sensibilité aux conditions initiales**, caractéristique de la série chaotique de Hénon.

## ▶️ Exécution

```bash
python henon_generator.py
python architecture_search.py
python training.py
python predictions.py
```

Pour consulter l'interface, ouvrir :

```text
interface/index.html
```

## 🎓 Contexte

**Mini-projet d'application 2026**

**Institut Supérieur Polytechnique de Madagascar — ESIIA 4**

**Auteur :** RAKOTOARISOA Heriniaina Steve

**Année universitaire :** 2025–2026

---

<p align="center">
  🧠 <strong>Neural Networks</strong> × 🌀 <strong>Chaos</strong> × 🐍 <strong>Python</strong>
</p>
