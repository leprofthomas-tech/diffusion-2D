import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Ajout pour backend interactif Qt
try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except NameError:
    # Si le code n'est pas exécuté dans IPython (Spyder/Jupyter), on ignore
    pass

# ================================
# 1. Paramètres de la grille
# ================================
N_cell = 11
Nx, Ny = N_cell, N_cell
dx = dy = 1.0
alpha = 1
dt = 0.25
steps = 201
CFL = alpha*dt*(1/dx**2 + 1/dy**2) # < 0.5
print(CFL)

# ================================
# 2. Paramètres de la source
# ================================
source_temp = 50      # température de la source
fix_source = True      # True = température constante, False = initiale seulement
source_type = 'center'  # 'center' ou 'column'
column_index = 0        # si 'column', quelle colonne chauffer

# ================================
# 3. Initialisation du champ
# ================================
T0 = np.zeros((Ny, Nx))             # champ initial de température
fixed_mask = np.zeros_like(T0, bool)  # masque des cellules à température fixe

# --- Source centrale ou colonne
if source_type == 'center':
    source_pos = (Ny//2, Nx//2)
    T0[source_pos] = source_temp
    fixed_mask[source_pos] = fix_source
elif source_type == 'column':
    T0[:, column_index] = source_temp
    fixed_mask[:, column_index] = fix_source

# Historique et pas courant
history = [T0.copy()]
current_step = 0

# ================================
# 4. Création de la figure
# ================================
if Nx < 6 and Ny < 6:
    fig = plt.figure(figsize=(12,6))

    # Axes pour la grille (décalée à gauche)
    ax = fig.add_axes([0.05, 0.1, 0.55, 0.8])

else:
    fig, ax = plt.subplots(figsize=(6,6))

im = ax.imshow(T0, cmap='hot', origin='lower', vmin=0, vmax=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Température")
ax.set_title("Diffusion 2D - schéma explicite (Forward Euler)")

# Texte pour le pas
time_text = ax.text(0.02, 1.02, '', transform=ax.transAxes,
                    fontsize=12, color='white',
                    bbox=dict(facecolor='black', alpha=0.5))

# Texte dans chaque cellule
text_grid = [[ax.text(j, i, '', color='white', ha='center', va='center')
              for j in range(Nx)] for i in range(Ny)]

# ================================
# 4b. Affichage des équations si petite grille
# ================================
if Nx < 6 and Ny < 6:
    ax_eq = fig.add_axes([0.60, 0.1, 0.35, 0.55])  # axes pour les équations
    ax_eq.axis('off')  # on n'affiche pas d'axes

    # 0) Équation générale de la diffusion (avec terme source)
    general_eq_full = r"$\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)$" # + \frac{Q}{\rho c_p}
    ax_eq.text(0, 1.0, general_eq_full, fontsize=16, color='black', ha='left', va='top')

    # 1) Équation discrète pour le schéma explicite
    general_eq = r"$\frac{T_{i,j}^{t+dt} - T_{i,j}^{t}}{\Delta t} = \alpha \left(\frac{T_{i+1,j}-2T_{i,j}+T_{i-1,j}}{\Delta x^2} + \frac{T_{i,j+1}-2T_{i,j}+T_{i,j-1}}{\Delta y^2}\right)$"
    ax_eq.text(0, 0.8, general_eq, fontsize=16, color='black', ha='left', va='top')

    # 2) Équation discrète pour T(i+dt)
    discrete_eq = r"$T_{i,j}^{t+dt} = T_{i,j}^{t} + \alpha \Delta t \left(\frac{T_{i+1,j}^{t}-2T_{i,j}^{t}+T_{i-1,j}^{t}}{\Delta x^2} + \frac{T_{i,j+1}^{t}-2T_{i,j}^{t}+T_{i,j-1}^{t}}{\Delta y^2}\right)$"
    ax_eq.text(0, 0.6, discrete_eq, fontsize=16, color='black', ha='left', va='top')

# ================================
# 5. Flux internes (si petite grille)
# ================================
show_flux = Nx < 6 and Ny < 6
if show_flux:
    flux_texts = []
    for i in range(Ny):
        for j in range(Nx):
            if j < Nx-1:
                flux_texts.append(ax.text(j+0.5, i, '', color='cyan', fontsize=8, ha='center', va='center'))
            if i < Ny-1:
                flux_texts.append(ax.text(j, i+0.5, '', color='lime', fontsize=8, ha='center', va='center'))

# ================================
# 6. Fonction pour le pas suivant
# ================================
def compute_next(T):
    T_new = T.copy()
    for i in range(Ny):
        for j in range(Nx):
            T_up    = T[i-1,j] if i>0 else T[i,j]
            T_down  = T[i+1,j] if i<Ny-1 else T[i,j]
            T_left  = T[i,j-1] if j>0 else T[i,j]
            T_right = T[i,j+1] if j<Nx-1 else T[i,j]
            T_new[i,j] = T[i,j] + alpha*dt*((T_up-2*T[i,j]+T_down)/dx**2 + (T_left-2*T[i,j]+T_right)/dy**2)
            # Schéma explicite (forward Euler) pour l'équation de diffusion 2D :
            # T_new[i,j] = T[i,j] + alpha*dt * (d²T/dx² + d²T/dy²)
            # 
            # Approximation des dérivées secondes par différences finies centrales :
            #   d²T/dx² ≈ (T[i+1,j] - 2*T[i,j] + T[i-1,j]) / dx²
            #   d²T/dy² ≈ (T[i,j+1] - 2*T[i,j] + T[i,j-1]) / dy²
            #
            # La température à l'instant suivant T_new[i,j] dépend donc de la valeur actuelle
            # du nœud et de ses 4 voisins immédiats (haut, bas, gauche, droite).
            # Ce schéma est simple à coder mais stable seulement si
            # alpha*dt*(1/dx² + 1/dy²) ≤ 0.5 (critère CFL).
            
    # --- Source ponctuelle / température fixée
    # Dans l'équation générale de diffusion de la chaleur, il y aurait normalement un terme
    # de source volumique Q/(rho*c_p) : ∂T/∂t = α*(d²T/dx² + d²T/dy²) + Q/(ρ c_p)
    # Ici, on ne considère pas de source distribuée continue.
    # La "source" est modélisée simplement en fixant la température de certaines cellules
    # (au centre ou dans une colonne). Cela remplace le terme Q/(ρ c_p) par une condition
    # de température imposée, ce qui est suffisant pour le code pédagogique.
    if fix_source:
        T_new[fixed_mask] = source_temp
    return T_new

# ================================
# 7. Calcul des flux internes
# ================================
def compute_internal_fluxes(T):
    qx = np.zeros((Ny, Nx-1))
    qy = np.zeros((Ny-1, Nx))
    for i in range(Ny):
        for j in range(Nx-1):
            qx[i,j] = -(T[i,j+1] - T[i,j]) / dx
    for i in range(Ny-1):
        for j in range(Nx):
            qy[i,j] = -(T[i+1,j] - T[i,j]) / dy
    return qx, qy

# ================================
# 8. Mise à jour de l'affichage
# ================================
def update_plot():
    T = history[current_step]
    im.set_array(T)
    time_text.set_text(f"Step: {current_step}")

    # Affichage des températures
    for i in range(Ny):
        for j in range(Nx):
            text_grid[i][j].set_text(f"{T[i,j]:.0f}")

    # Affichage des flux si grille petite
    if show_flux:
        qx, qy = compute_internal_fluxes(T)
        k = 0
        for i in range(Ny):
            for j in range(Nx):
                if j < Nx-1:
                    flux_texts[k].set_text(f"{qx[i,j]:.0f}")
                    k += 1
                if i < Ny-1:
                    flux_texts[k].set_text(f"{qy[i,j]:.0f}")
                    k += 1
    fig.canvas.draw_idle()

# ================================
# 9. Gestion clavier (gauche/droite)
# ================================
def on_key(event):
    global current_step
    if event.key == 'right':
        if current_step < len(history)-1:
            current_step += 1
        elif current_step < steps-1:
            next_T = compute_next(history[-1])
            history.append(next_T)
            current_step += 1
        update_plot()
    elif event.key == 'left':
        if current_step > 0:
            current_step -= 1
            update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)

# ================================
# 10. Affichage initial
# ================================
update_plot()
plt.show()
