# Diffusion 2D de la chaleur

Auteur : Thomas Elisabettini  
Date : 29/10/2025  

---

## Description

Ce projet contient un **code pédagogique en Python** illustrant la **diffusion de la chaleur en 2D** avec :

- Une grille Nx × Ny de cellules.  
- Une **source ponctuelle** (au centre ou dans une colonne) dont la température peut être fixée.  
- Affichage de la **température de chaque cellule** et, pour les petites grilles (Nx, Ny < 6), des **flux internes et équations**.  
- Visualisation interactive : navigation dans le temps avec les touches **flèche gauche / droite**.  

Le code utilise **NumPy** et **Matplotlib**.

---

## Lancement du code

1. Assurez-vous d’avoir installé les dépendances :

```bash
pip install numpy matplotlib
=======
# diffusion-2D
Simulation pédagogique de la diffusion de la chaleur en 2D avec visualisation des températures, flux et équations discrètes.
