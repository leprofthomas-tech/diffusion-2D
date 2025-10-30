import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ================================
# 1. Paramètres de la grille
# ================================
N_cel = 11
Nx, Ny = N_cel, N_cel
dx = dy = 1.0
alpha = 1.0
dt = 2.0
steps = 201

# ================================
# 2. Paramètres de la source
# ================================
source_temp = 40
fix_source = True
source_type = 'center'
column_index = 0

# ================================
# 3. Initialisation du champ
# ================================
T0 = np.zeros((Ny, Nx))
fixed_mask = np.zeros_like(T0, dtype=bool)

if source_type == 'center':
    source_pos = (Ny//2, Nx//2)
    T0[source_pos] = source_temp
    fixed_mask[source_pos] = fix_source
elif source_type == 'column':
    T0[:, column_index] = source_temp
    fixed_mask[:, column_index] = fix_source

history = [T0.copy()]
current_step = 0

# ================================
# 4. Construction Laplacien discret
# ================================
N = Nx * Ny
def idx(i,j): return i*Nx + j

def build_laplacian(Nx, Ny, dx, dy):
    N = Nx*Ny
    L = sparse.lil_matrix((N,N), dtype=float)
    for i in range(Ny):
        for j in range(Nx):
            k = idx(i,j)
            # x
            if i>0:   L[k, idx(i-1,j)] += 1/dx**2
            if i<Ny-1: L[k, idx(i+1,j)] += 1/dx**2
            # y
            if j>0:   L[k, idx(i,j-1)] += 1/dy**2
            if j<Nx-1: L[k, idx(i,j+1)] += 1/dy**2
            # diag
            diag = 0.0
            if i>0: diag += 1/dx**2
            if i<Ny-1: diag += 1/dx**2
            if j>0: diag += 1/dy**2
            if j<Nx-1: diag += 1/dy**2
            L[k,k] = -diag
    return L.tocsr()

Lap = build_laplacian(Nx, Ny, dx, dy)
A_sys = sparse.identity(N) - alpha*dt*Lap

# ================================
# 5. Utilitaires indexation
# ================================
def flatten(T): return T.ravel()
def unflatten(v): return v.reshape((Ny, Nx))

# ================================
# 6. Conditions Dirichlet
# ================================
def apply_dirichlet(A_csr, b, fixed_mask, T_fixed_value):
    A = A_csr.tocsr(copy=True)
    b = b.copy()
    fixed_indices = np.where(fixed_mask.ravel())[0]
    for j in fixed_indices:
        Tj = T_fixed_value if np.isscalar(T_fixed_value) else T_fixed_value.ravel()[j]
        col_j = A.getcol(j).toarray().ravel()
        b -= col_j*Tj
        A = A.tolil()
        A[:, j] = 0.0
        A[j, :] = 0.0
        A[j, j] = 1.0
        A = A.tocsr()
        b[j] = Tj
    return A, b

# ================================
# 7. Flux internes
# ================================
def compute_internal_fluxes(T):
    qx = np.zeros((Ny, Nx-1))
    qy = np.zeros((Ny-1, Nx))
    for i in range(Ny):
        for j in range(Nx-1): qx[i,j] = -(T[i,j+1]-T[i,j])/dx
    for i in range(Ny-1):
        for j in range(Nx): qy[i,j] = -(T[i+1,j]-T[i,j])/dy
    return qx,qy

# ================================
# 8. Schéma implicite
# ================================
def compute_next_implicit(T):
    Tn = flatten(T)
    b = Tn.copy()
    A_mod, b_mod = apply_dirichlet(A_sys, b, fixed_mask, source_temp)
    Tnp1_flat = spsolve(A_mod, b_mod)
    Tnp1 = unflatten(Tnp1_flat)
    Tnp1[fixed_mask] = source_temp
    return Tnp1

# ================================
# 9. Affichage
# ================================
if Nx<6 and Ny<6:
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_axes([0.05, 0.1, 0.55, 0.8])
else:
    fig, ax = plt.subplots(figsize=(6,6))

im = ax.imshow(T0, cmap='hot', origin='lower', vmin=0, vmax=100)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Température")
ax.set_title("Diffusion 2D - schéma implicite (Backward Euler)")
time_text = ax.text(0.02, 1.02, '', transform=ax.transAxes, fontsize=12, color='white',
                    bbox=dict(facecolor='black', alpha=0.5))
text_grid = [[ax.text(j,i,'', color='white', ha='center', va='center') for j in range(Nx)] for i in range(Ny)]

show_flux = Nx<6 and Ny<6
if show_flux:
    flux_texts=[]
    for i in range(Ny):
        for j in range(Nx):
            if j<Nx-1: flux_texts.append(ax.text(j+0.5,i,'', color='cyan', fontsize=8, ha='center', va='center'))
            if i<Ny-1: flux_texts.append(ax.text(j,i+0.5,'', color='lime', fontsize=8, ha='center', va='center'))

# ================================
# 10. Équations pédagogiques
# ================================
if Nx<6 and Ny<6:
    ax_eq = fig.add_axes([0.60, 0.1, 0.35, 0.55])
    ax_eq.axis('off')
    ax_eq.text(0,1.0,r"$\frac{\partial T}{\partial t} = \alpha (\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2})$", fontsize=16, ha='left')
    ax_eq.text(0,0.8,r"$\frac{T_{i,j}^{t+dt}-T_{i,j}^{t}}{\Delta t} = \alpha(\frac{T_{i+1,j}^{t+dt}-2T_{i,j}^{t+dt}+T_{i-1,j}^{t+dt}}{\Delta x^2} + \frac{T_{i,j+1}^{t+dt}-2T_{i,j}^{t+dt}+T_{i,j-1}^{t+dt}}{\Delta y^2})$", fontsize=16, ha='left')
    backward_euler_eq = (
        r"$T_{i,j}^{t+dt} = T_{i,j}^{t} + \alpha \Delta t "
        r"\left("
        r"\frac{T_{i+1,j}^{t+dt} - 2T_{i,j}^{t+dt} + T_{i-1,j}^{t+dt}}{\Delta x^2} + "
        r"\frac{T_{i,j+1}^{t+dt} - 2T_{i,j}^{t+dt} + T_{i,j-1}^{t+dt}}{\Delta y^2}"
        r"\right)$"
    )
    ax_eq.text(0, 0.7, backward_euler_eq, fontsize=16, color='black', ha='left', va='top')
    #ax_eq.text(0,0.6,r"$T_{i,j}^{t+dt} = T_{i,j}^{t} + \alpha \Delta t(...\,)$", fontsize=16, ha='left')
    ax_eq.text(0,0.45,r"$(I - \alpha \Delta t\,L) T^{t+dt} = T^t$", fontsize=16, ha='left')

# ================================
# 11. Affichage dynamique du système linéaire
# ================================
if Nx<6 and Ny<6:
    ax_sys = fig.add_axes([0.60,0.7,0.35,0.25])
    ax_sys.axis('off')
    system_text = ax_sys.text(0,1.0,"", fontsize=12, color='navy', ha='left', va='top', family='monospace')

    def update_system_text(Tn):
        A = (sparse.identity(N) - alpha*dt*Lap).toarray()
        A_str = np.array2string(A, formatter={'float_kind':lambda x:f"{x:5.0f}"})
        b_str = np.array2string(Tn.ravel(), formatter={'float_kind':lambda x:f"{x:5.1f}"})
        system_text.set_text(r"$A\,T^{t+Δt}=T^{t}$" + "\n" + f"A = (I - αΔt L)\n{A_str}\n\nT^t = {b_str}")

# ================================
# 12. Fonction update_plot
# ================================
def update_plot():
    T = history[current_step]
    im.set_array(T)
    time_text.set_text(f"Step: {current_step}")

    # afficher températures dans chaque cellule seulement si grille pas trop grande
    if Nx <= 20 and Ny <= 20:
        for i in range(Ny):
            for j in range(Nx):
                text_grid[i][j].set_text(f"{T[i,j]:.0f}")
    else:
        # vider le texte si on dépasse la taille limite
        for i in range(Ny):
            for j in range(Nx):
                text_grid[i][j].set_text("")

    # flux internes si petite grille
    if show_flux:
        qx,qy = compute_internal_fluxes(T)
        k=0
        for i in range(Ny):
            for j in range(Nx):
                if j<Nx-1: flux_texts[k].set_text(f"{qx[i,j]:.0f}"); k+=1
                if i<Ny-1: flux_texts[k].set_text(f"{qy[i,j]:.0f}"); k+=1

    # texte système linéaire si petite grille
    if Nx<6 and Ny<6:
        update_system_text(T)

    fig.canvas.draw_idle()


# ================================
# 13. Gestion clavier
# ================================
def on_key(event):
    global current_step
    if event.key=='right':
        if current_step < len(history)-1:
            current_step +=1
        elif current_step<steps-1:
            next_T = compute_next_implicit(history[-1])
            history.append(next_T)
            current_step +=1
        update_plot()
    elif event.key=='left':
        if current_step>0:
            current_step-=1
            update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)

# ================================
# 14. Affichage initial
# ================================
update_plot()
plt.show()
