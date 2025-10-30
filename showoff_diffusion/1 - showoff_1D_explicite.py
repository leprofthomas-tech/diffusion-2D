import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Backend interactif (Spyder/Jupyter)
try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except NameError:
    pass

# ================================
# 1. Paramètres de la grille
# ================================
N_cell = 5
Nx, Ny = N_cell, 1
dx = dy = 1.0
alpha = 1
dt = 0.25
steps = 201
CFL = alpha * dt * (1 / dx ** 2 + 1 / dy ** 2)
print("CFL =", CFL)

# ================================
# 2. Paramètres de la source
# ================================
source_temp = 50
fix_source = True
source_type = 'column'
column_index = 0

# ================================
# 3. Initialisation
# ================================
T0 = np.zeros((Ny, Nx))
fixed_mask = np.zeros_like(T0, bool)

if source_type == 'center':
    source_pos = (Ny // 2, Nx // 2)
    T0[source_pos] = source_temp
    fixed_mask[source_pos] = fix_source
elif source_type == 'column':
    T0[:, column_index] = source_temp
    fixed_mask[:, column_index] = fix_source

history = [T0.copy()]
current_step = 0

# ================================
# 4. Création de la figure
# ================================
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0.05, 0.1, 0.5, 0.8])  # largeur réduite pour laisser de la place aux équations

# Affichage de la grille
im = ax.imshow(T0, cmap='hot', origin='lower', vmin=0, vmax=100)

# Colorbar proprement positionnée
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("Température")

ax.set_title("Diffusion 1D - schéma explicite (Forward Euler)")

time_text = ax.text(0.02, 1.02, '', transform=ax.transAxes,
                    fontsize=12, color='white',
                    bbox=dict(facecolor='black', alpha=0.5))

text_grid = [[ax.text(j, i, '', color='white', ha='center', va='center')
              for j in range(Nx)] for i in range(Ny)]

# ================================
# 4b. Affichage des équations
# ================================
if Nx < 6 and Ny < 6:
    ax_eq = fig.add_axes([0.65, 0.1, 0.35, 0.55])
    ax_eq.axis('off')

    general_eq_full = r"$\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)$"
    ax_eq.text(0, 1.0, general_eq_full, fontsize=16, color='black', ha='left', va='top')

    general_eq = r"$\frac{T_{i}^{t+dt} - T_{i}^{t}}{\Delta t} = \alpha \left(\frac{T_{i+1}-2T_{i}+T_{i-1}}{\Delta x^2}\right)$"
    ax_eq.text(0, 0.8, general_eq, fontsize=16, color='black', ha='left', va='top')

    discrete_eq = r"$T_{i}^{t+dt} = T_{i}^{t} + \alpha \Delta t \left(\frac{T_{i+1}^{t}-2T_{i}^{t}+T_{i-1}^{t}}{\Delta x^2}\right)$"
    ax_eq.text(0, 0.6, discrete_eq, fontsize=16, color='black', ha='left', va='top')

# ================================
# 5. Flux internes
# ================================
show_flux = Nx < 6 and Ny < 6
if show_flux:
    flux_texts = []
    for i in range(Ny):
        for j in range(Nx):
            if j < Nx - 1:
                flux_texts.append(ax.text(j + 0.5, i, '', color='cyan', fontsize=8, ha='center', va='center'))
            if i < Ny - 1:
                flux_texts.append(ax.text(j, i + 0.5, '', color='lime', fontsize=8, ha='center', va='center'))

# ================================
# 5b. Zone pour les calculs numériques
# ================================
ax_calc = fig.add_axes([0.05, 0.02, 0.9, 0.25])
ax_calc.axis('off')
calc_texts = []
for j in range(Nx):
    txt = ax_calc.text(0, 1 - j * 0.18, "", fontsize=12, va='top', ha='left', color='navy', family='monospace')
    calc_texts.append(txt)

# ================================
# 6. Fonction pour le pas suivant
# ================================
def compute_next(T):
    T_new = T.copy()
    for i in range(Ny):
        for j in range(Nx):
            T_up = T[i - 1, j] if i > 0 else T[i, j]
            T_down = T[i + 1, j] if i < Ny - 1 else T[i, j]
            T_left = T[i, j - 1] if j > 0 else T[i, j]
            T_right = T[i, j + 1] if j < Nx - 1 else T[i, j]
            T_new[i, j] = T[i, j] + alpha * dt * (
                (T_up - 2 * T[i, j] + T_down) / dx ** 2 +
                (T_left - 2 * T[i, j] + T_right) / dy ** 2
            )
    if fix_source:
        T_new[fixed_mask] = source_temp
    return T_new

# ================================
# 7. Calcul des flux internes
# ================================
def compute_internal_fluxes(T):
    qx = np.zeros((Ny, Nx - 1))
    qy = np.zeros((Ny - 1, Nx))
    for i in range(Ny):
        for j in range(Nx - 1):
            qx[i, j] = -(T[i, j + 1] - T[i, j]) / dx
    for i in range(Ny - 1):
        for j in range(Nx):
            qy[i, j] = -(T[i + 1, j] - T[i, j]) / dy
    return qx, qy

# ================================
# 8. Mise à jour de l'affichage
# ================================
def update_plot():
    T = history[current_step]
    im.set_array(T)
    time_text.set_text(f"Step: {current_step}")

    for i in range(Ny):
        for j in range(Nx):
            text_grid[i][j].set_text(f"{T[i, j]:.0f}")

    if show_flux:
        qx, qy = compute_internal_fluxes(T)
        k = 0
        for i in range(Ny):
            for j in range(Nx):
                if j < Nx - 1:
                    flux_texts[k].set_text(f"{qx[i, j]:.0f}")
                    k += 1
                if i < Ny - 1:
                    flux_texts[k].set_text(f"{qy[i, j]:.0f}")
                    k += 1

    # Calcul numérique pour toutes les cellules
    for j in range(Nx):
        i = 0  # 1D ici
        if fixed_mask[i, j]:   # 🔹 cellule source
            calc_texts[j].set_text(f"T_new[{i},{j}] = {source_temp:.2f}")
        else:
            T_up = T[i - 1, j] if i > 0 else T[i, j]
            T_down = T[i + 1, j] if i < Ny - 1 else T[i, j]
            T_left = T[i, j - 1] if j > 0 else T[i, j]
            T_right = T[i, j + 1] if j < Nx - 1 else T[i, j]
    
            delta = alpha * dt * (
                (T_up - 2 * T[i, j] + T_down) / dx ** 2 +
                (T_left - 2 * T[i, j] + T_right) / dy ** 2
            )
            T_new_val = T[i, j] + delta
    
            calc_texts[j].set_text(
                f"T_step{current_step+1}[{j}] = {T[i,j]:.2f} + {alpha}*{dt}"
                f"*(({T_up:.2f}-2*{T[i,j]:.2f}+{T_down:.2f})/{dx**2}"
                f" + ({T_left:.2f}-2*{T[i,j]:.2f}+{T_right:.2f})/{dy**2}) = {T_new_val:.2f}"
            )


    fig.canvas.draw_idle()

# ================================
# 9. Gestion clavier
# ================================
def on_key(event):
    global current_step
    if event.key == 'right':
        if current_step < len(history) - 1:
            current_step += 1
        elif current_step < steps - 1:
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
