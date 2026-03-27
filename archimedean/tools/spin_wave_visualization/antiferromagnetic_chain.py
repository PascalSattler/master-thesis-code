import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# get lmodern font from LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{lmodern}"
})

# dispersion function
def dispersion(k_value:float, lattice_const:float = 1, exchange:float = -1, spin:float = 1, anisotropy:float = 0, B_field:float = 0) -> float:
    if anisotropy == 0 and B_field == 0:
        energy:float = 2*np.abs(exchange)*spin*np.abs(np.sin(k_value*lattice_const))
        return energy
    else:
        energy:float = 2*np.abs(exchange)*spin*np.sqrt((1+anisotropy/exchange)**2 - np.cos(k_value*lattice_const)**2)
        return energy + B_field, energy - B_field
    
# sublattice localization function
def probability(k_value:float, lattice_const:float = 1, exchange:float = -1, spin:float = 1, anisotropy:float = 0, B_field:float = 0) -> float:
    if anisotropy == 0 and B_field == 0:
        prob:float = np.abs(np.sin(k_value*lattice_const))
        return 0.5*(1+prob), 0.5*(1-prob)
    else:
        gamma1:float = 1 + anisotropy/exchange + B_field/(np.abs(exchange)*spin)
        gamma2:float = 1 + anisotropy/exchange - B_field/(np.abs(exchange)*spin)
        prob1:float = 1/(1+np.abs(np.cos(k_value*lattice_const)/(np.sqrt((1+anisotropy/exchange)**2 - np.cos(k_value*lattice_const)**2)+gamma1))**2)
        prob2:float = 1/(1+np.abs((np.sqrt((1+anisotropy/exchange)**2 - np.cos(k_value*lattice_const)**2)+gamma2)/np.cos(k_value*lattice_const))**2)
        return prob1, 1-prob1

# parameters
J = -1
A = -0.01
B = 0.2

# k values
N_points = 1001
midpoint = N_points // 2
k_values = np.linspace(-np.pi/2, np.pi/2, N_points)

# plot 1 and 2 graph setup
branch_p1 = np.empty(len(k_values))
probs1_p1 = np.empty(len(k_values))
probs2_p1 = np.empty(len(k_values))

branch1_p2 = np.empty(len(k_values))
branch2_p2 = np.empty(len(k_values))
probs1_p2 = np.empty(len(k_values))
probs2_p2 = np.empty(len(k_values))
compare_graph_p2 = np.empty(len(k_values))

# calculate dispersion and sublattice localization
for i,j in enumerate(k_values):
    branch_p1[i] = dispersion(j)
    probs1_p1[i], probs2_p1[i] = probability(j)

    branch1_p2[i], branch2_p2[i]= dispersion(j, anisotropy=A, B_field=B)
    probs1_p2[i], probs2_p2[i] = probability(j, anisotropy=A, B_field=B)#
    compare_graph_p2[i], no_use= dispersion(j, anisotropy=A)

# plot layout setup
fig, axs = plt.subplots(ncols=2, sharey=True, constrained_layout=True, figsize = (8,4))

marker_kwargs = dict(marker = '.', s=15, edgecolors='none', cmap='coolwarm', vmin=0, vmax=1)

# plot 1 graph
plot1 = axs[0].scatter(k_values[midpoint:], branch_p1[midpoint:], c=probs1_p1[midpoint:], **marker_kwargs)
plot1 = axs[0].scatter(k_values[:midpoint], branch_p1[:midpoint], c=probs2_p1[:midpoint], **marker_kwargs)

# plot 2 graph
plot2 = axs[1].scatter(k_values, branch1_p2, c=probs1_p2, **marker_kwargs)
plot2 = axs[1].scatter(k_values, branch2_p2, c=probs2_p2, **marker_kwargs)
plot2 = axs[1].plot(k_values, compare_graph_p2, linestyle = 'dashed', linewidth = 1.2, color = 'lightgrey')

#markers
# plot1 = axs[0].scatter(np.pi/2, dispersion(np.pi/2), facecolors = 'none', edgecolors= 'black', marker = 'o', linewidths = 1, s = 50)

# axis labels
axs[0].set_xlabel(r'$ka$', fontsize = 14)
axs[0].set_ylabel(r'$\varepsilon(k)/(|J|\hbar^2 S)$', fontsize = 14)
axs[1].set_xlabel(r'$ka$', fontsize = 14)

# x axis style
xtick_positions = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]
xtick_labels = [r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"]
axs[0].set_xticks(xtick_positions, xtick_labels)
axs[1].set_xticks(xtick_positions, xtick_labels)
axs[0].tick_params(labelsize=14)
axs[1].tick_params(labelsize=14)

# y axis style
ytick_positions = [0,0.5,1,1.5,2]
ytick_labels = ["0","","1","","2"]
axs[0].set_yticks(ytick_positions, ytick_labels)
axs[1].set_yticks(ytick_positions, ytick_labels)
axs[1].tick_params(labelleft=True)
axs[1].set_ylim(0, max(max(branch1_p2),max(branch2_p2))+0.1)

# grid style
axs[0].grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.4)
axs[0].set_axisbelow(True)
axs[1].grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.4)
axs[1].set_axisbelow(True)

# legend setup
cmap = plt.get_cmap('coolwarm')
norm = Normalize(vmin=0, vmax=1)
color_min = cmap(norm(0))
color_max = cmap(norm(1))
axs[0].plot([], [], color=color_max, lw=1, label = r"$\varepsilon_1(k>0)$")
axs[0].plot([], [], color=color_min, lw=1, label = r"$\varepsilon_2(k<0)$")
axs[1].plot([], [], color=color_max, lw=1, label = r"$\varepsilon_1$")
axs[1].plot([], [], color=color_min, lw=1, label = r"$\varepsilon_2$")
axs[0].legend(title=r'(a) $A, B_z = 0$', loc="upper center", fontsize = 12) 
axs[1].legend(title=r'(b) $A, B_z \neq 0$', loc="upper center", fontsize = 12)

# margins
axs[0].margins(x=0, y=0)
axs[1].margins(x=0, y=0)

# colorbar style
cbar = fig.colorbar(plot1, ax=axs, pad = 0.02)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['0', '1'])
cbar.set_label('localization on q=1 (spin-up sublattice)', rotation=90, fontsize = 12)
cbar.ax.yaxis.set_label_coords(1.5, 0.5)
cbar.ax.tick_params(labelsize = 12)

#markers
plot1 = axs[0].scatter(np.pi/2, dispersion(np.pi/2), facecolors = 'none', edgecolors= 'black', marker = 'o', linewidths = 1, s = 50)
plot1 = axs[0].text(0.85*np.pi/2, 1.05*dispersion(np.pi/2), r"$(c)$", fontsize = 12)
plot1 = axs[0].scatter(np.pi/4, dispersion(np.pi/4), facecolors = 'none', edgecolors= 'black', marker = 'o', linewidths = 1, s = 50)
plot1 = axs[0].text(1.15*np.pi/4, 0.95*dispersion(np.pi/4), r"$(d)$", fontsize = 12)
plot1 = axs[0].scatter(-np.pi/4, dispersion(-np.pi/4), facecolors = 'none', edgecolors= 'black', marker = 'o', linewidths = 1, s = 50)
plot1 = axs[0].text(-1.4*np.pi/4, 0.95*dispersion(-np.pi/4), r"$(e)$", fontsize = 12)
plot1 = axs[0].scatter(-np.pi/2, dispersion(-np.pi/2), facecolors = 'none', edgecolors= 'black', marker = 'o', linewidths = 1, s = 50)
plot1 = axs[0].text(-0.975*np.pi/2, 1.05*dispersion(-np.pi/2), r"$(f)$", fontsize = 12)

plt.savefig('plots_from_analytics/antiferromagnet_dispersion.pdf', facecolor='w', transparent=False, dpi = 1000)

print(color_max, color_min)