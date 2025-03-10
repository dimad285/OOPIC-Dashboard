import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import h5py

kb = 8.617e-5 #1.38064852e-23  # Boltzmann constant
kb_j = 1.38064852e-23
qe = -1.60217662e-19  # Elementary charge    
mp = 1.6726219e-27  # Mass of hydrogen
e0 = 8.85418782e-12  # Vacuum permittivity
me = 9.1e-31
T_ratio = 11604.518


def read_particles_as_matrix(dump_file_path, particle_name: str):

    with h5py.File(dump_file_path, 'r') as f:
        if particle_name not in f:
            return np.empty((5, 0))
        elif len(f[particle_name]) == 0:
            return np.empty((5, 0))

        data_list = []
        for group_name in f[particle_name].keys():
            group = f[particle_name][group_name]
            if 'ptcls' in group:
                data = group['ptcls'][()]
                if data.shape[1] == 5:
                    data_list.append(data)
        
        
        particle_data = np.vstack(data_list).T

    return particle_data
    
def read_np2c(dump_file_path, particle_name: str):

    with h5py.File(dump_file_path, 'r') as f:
        if particle_name not in f:
            return 1
        elif len(f[particle_name]) == 0:
            return 1

        
        np2c = f[particle_name]['pGrp0'].attrs['np2c'][0]

    return np2c


def read_grid_dimension(dump_file_path):

    with h5py.File(dump_file_path, 'r') as f:
        
        z, r = f['SpatialRegion'].attrs['dims'][2:]
        m, n = f['NGD0']['NGD'].shape[:2]
        

def get_cell_indicies(particle_data, Z, R, m, n):

    '''
    For all particles in particle_data calculate cell index where it resides

    Parameters:
    - particle_data: (5, n) 
    - Z: Total domain height.
    - R: Maximum radial distance.
    - m: Number of grid points in z-direction.
    - n: Number of grid points in r-direction.
    '''

    cell_indices = np.zeros((2, particle_data.shape[1]), dtype=np.int32)

    if particle_data.shape[1] == 0:
        print('0 particles')
        return cell_indices

    dz = Z / (m - 1)
    dr = R / (n - 1)

    # Normalize particle positions to grid
    x_norm = particle_data[0, :] / dz
    y_norm = particle_data[1, :] / dr

    # Find the cell index for each particle
    cell_indices[0, :] = np.floor(x_norm).astype(int)
    cell_indices[1, :] = np.floor(y_norm).astype(int) 

    return cell_indices


def get_numerical_density(cell_indices, Z, R, m, n, np2c = 1):
    """
    Compute numerical density in a 2D axisymmetric cylindrical grid.

    Parameters:
    - cell_indices: (2, n_particles) array containing (z_idx, r_idx).
    - Z: Total domain height.
    - R: Maximum radial distance.
    - m: Number of grid points in z-direction.
    - n: Number of grid points in r-direction.

    Returns:
    - nd: (m-1, n-1) array of numerical density (particles per unit volume).
    """

    dz = Z / (m - 1)  # Axial cell spacing
    dr = R / (n - 1)  # Radial cell spacing

    # Compute radial positions (excluding last point)
    r_array = np.linspace(0, R - dr, n - 1)

    # Compute differential volume: dV = (dS * dz), where dS is annular area
    dS = np.pi * ((r_array + dr)**2 - r_array**2)  # Annular ring area
    dV = dS * dz  # Element volume in cylindrical coords

    # Initialize density array
    nd = np.zeros((m-1, n-1), dtype=float)

    # Count particles in each (z, r) cell
    np.add.at(nd, (cell_indices[0], cell_indices[1]), 1)

    # Normalize by cell volume
    nd /= dV[None, :]  # Broadcast across all z-layers

    return nd * np2c


def get_energy_distribution(particle_data, M, bins = 64):

    if particle_data.shape[1] == 0:
        print('0 particles')
        return (np.arange(bins), np.zeros(bins))
    
    v = particle_data[2:, :]
    E = (np.sum(v**2, axis=0)) * M * 0.5 / (-qe)
    x = np.arange(bins) * np.max(E)/bins
    return (x, np.histogram(E, bins, density=True)[0])

def get_velocity_distribution(particle_data, bins = 64):

    if particle_data.shape[1] == 0:
        print('0 particles')
        return (np.arange(bins), np.zeros(bins))
    
    vz, vr, vf = particle_data[2:, :]
    v = np.sqrt(vz**2 + vr**2 + vf**2)
    x = np.arange(bins) * np.max(v)/bins

    return (x, np.histogram(v, bins, density=True)[0])

def get_plasma_temperature(particle_data, M):

    if particle_data.shape[1] == 0:
        print('0 particles')
        return 0
    
    v = particle_data[2:, :]
    e_mean = np.mean(np.sum(v**2, axis=0)) * M / 2 / (-qe)
    return 2 * e_mean / (3 * kb) / T_ratio

def get_temperature_distribution(particle_data, cell_indices, m, n, M):
    """
    Computes the temperature distribution over a 2D cylindrical grid.

    Parameters:
    - particle_speeds: (n_particles,) array of speeds |v| of each particle.
    - cell_indices: (2, n_particles) array containing (z_idx, r_idx) where each particle resides.
    - grid_shape: (m-1, n-1) tuple for the temperature grid.
    - particle_mass: Mass of a single particle.
    - kB: Boltzmann constant.

    Returns:
    - T: (m-1, n-1) array of temperature values.
    """

    
    
    # Initialize temperature and particle count arrays
    T = np.zeros((m-1, n-1), dtype=float)

    if particle_data.shape[1] == 0:
        print('0 particles')
        return T
    
    particle_count = np.zeros((m-1, n-1), dtype=int)

    # Compute squared speeds
    v = particle_data[2:, :]
    v_squared = np.sum(v**2, axis=0)
    E = v_squared * M * 0.5 / (-qe)

    # Accumulate sum of squared speeds per cell
    np.add.at(T, (cell_indices[0], cell_indices[1]), E)

    # Count the number of particles in each cell
    np.add.at(particle_count, (cell_indices[0], cell_indices[1]), 1)

    # Avoid division by zero
    mask = particle_count > 0
    T[mask] = (T[mask] / particle_count[mask]) * 2 / (3 * kb)

    return T / T_ratio


def get_plasma_density(numerical_density, q):
    return numerical_density * q



def get_debye_length(ne, ni, Te, Ti):

    a = e0 * kb / qe**2
    #Te == 0
    #mask_ti = Ti == 0
    #te = Te
    #ti = Ti
    #te[mask_te] = 1
    #ti[mask_ti] = 1

    c = 1
    d = 1
    b = (c + d)

    l_sq = a / b

    return np.sqrt(l_sq)


def get_distribution_function_E(particle_data, cell_indices, M, m, n, bins):

    """
    Compute velocity distributions in each grid cell.
    
    Parameters:
    - velocities: np.array of shape (2, n) -> (vx, vy) components of each particle
    - cell_indices: np.array of shape (2, n) -> (i, j) cell indices where each particle resides
    - m, n: int -> Number of cells in each axis (grid size)
    - bins: int -> Number of velocity bins
    
    Returns:
    - v_bins: np.array of shape (bins,) -> Bin centers for velocity magnitude
    - hist: np.array of shape (m, n, bins) -> Velocity distribution in each cell
    """

    if particle_data.shape[1] == 0:
        print('0 particles')
        return (None, None)
    
    v = particle_data[2:, :]
    # Compute velocity magnitudes
    v_sq = v[0]**2 + v[1]**2 + v[2]**2

    E = 0.5 * v_sq * M / (-qe)

    # Prepare histogram bins
    e_min, e_max = 0, np.max(E)
    e_bins = np.linspace(e_min, e_max, bins+1)
    bin_centers = 0.5 * (e_bins[:-1] + e_bins[1:])  # Get bin centers
    
    # Storage for velocity histograms in each cell
    hist = np.zeros((m, n, bins))

    # Populate histograms per cell
    for e, (i, j) in zip(E, cell_indices.T):
        bin_idx = np.searchsorted(e_bins, e, side="right") - 1  # Find bin index
        if 0 <= i < m and 0 <= j < n and 0 <= bin_idx < bins:
            hist[i, j, bin_idx] += 1

    # Normalize each cell histogram to a probability distribution
    hist /= np.sum(hist, axis=-1, keepdims=True, where=(hist > 0))

    return bin_centers, hist

def get_distribution_function_V(particle_data, cell_indices, m, n, bins):

    """
    Compute velocity distributions in each grid cell.
    
    Parameters:
    - velocities: np.array of shape (2, n) -> (vx, vy) components of each particle
    - cell_indices: np.array of shape (2, n) -> (i, j) cell indices where each particle resides
    - m, n: int -> Number of cells in each axis (grid size)
    - bins: int -> Number of velocity bins
    
    Returns:
    - v_bins: np.array of shape (bins,) -> Bin centers for velocity magnitude
    - hist: np.array of shape (m, n, bins) -> Velocity distribution in each cell
    """

    if particle_data.shape[1] == 0:
        print('0 particles')
        return (None, None)
    
    v = particle_data[2:, :]
    # Compute velocity magnitudes
    v_magnitudes = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

    # Prepare histogram bins
    v_min, v_max = 0, np.max(v_magnitudes)
    v_bins = np.linspace(v_min, v_max, bins+1)
    bin_centers = 0.5 * (v_bins[:-1] + v_bins[1:])  # Get bin centers
    
    # Storage for velocity histograms in each cell
    hist = np.zeros((m, n, bins))

    # Populate histograms per cell
    for v, (i, j) in zip(v_magnitudes, cell_indices.T):
        bin_idx = np.searchsorted(v_bins, v, side="right") - 1  # Find bin index
        if 0 <= i < m and 0 <= j < n and 0 <= bin_idx < bins:
            hist[i, j, bin_idx] += 1

    # Normalize each cell histogram to a probability distribution
    hist /= np.sum(hist, axis=-1, keepdims=True, where=(hist > 0))

    return bin_centers, hist



def plot_distribution(ax, v_bins, hist, i, j):
    """
    Plots the velocity distribution for a selected cell (i, j).

    Parameters:
    - v_bins: np.array of shape (bins,) -> Bin centers for velocity magnitude
    - hist: np.array of shape (m, n, bins) -> Velocity distributions in each cell
    - i, j: int -> Cell indices to visualize
    """
    distribution = hist[i, j]  # Get histogram for selected cell

    ax.plot(v_bins, distribution, marker='o', linestyle='-', label=f'Cell ({i}, {j})')
    plt.xlabel("Velocity Magnitude")
    plt.ylabel("Probability Density")
    plt.title("Velocity Distribution in Cell")
    plt.legend()
    plt.grid(True)
    plt.show()



def get_z_distribution(particle_data, Z, bins):

    if particle_data.shape[1] == 0:
        print('0 particles')
        return (np.arange(bins), np.zeros(bins))
    
    z = particle_data[0, :]
    x = np.arange(bins) * Z/bins
    return (x, np.histogram(z, bins, density=True)[0])


def get_r_distribution(particle_data, R, bins):

    if particle_data.shape[1] == 0:
        print('0 particles')
        return (np.arange(bins), np.zeros(bins))
    
    z = particle_data[1, :]
    x = np.arange(bins) * R/bins
    return (x, np.histogram(z, bins, density=True)[0])



class PlotApp:
    def __init__(self, root, ions_name, file_path, m, n, z, r, bins, boltzman = False, np2c = 1):
        self.root = root
        self.root.title("Plot Selector")
        #self.root.configure(bg='#24292e')
        self.dump_file = file_path
        self.m = m
        self.n = n
        self.z = z
        self.r = r
        self.bins = bins
        self.boltzman = boltzman
        self.ions_name = ions_name
        self.np2c = np2c

        # Dropdown menu
        self.plot_types = [
        "Te_heatmap", "Ti_heatmap", 
        "ne_heatmap", "ni_heatmap", 
        "rhoe_heatmap", "rhoi_heatmap", 
        "Te_surface", "Ti_surface", 
        "ne_surface", "ni_surface", 
        "rhoe_surface", "rhoi_surface",
        "e_velocity_distribution", "i_velocity_distribution",
        "e_energy_distribution", "i_energy_distribution",
        "debye_heatmap", "debye_surface",
        "zi_distribution", "ri_distribution",
        "ze_distribution", "re_distribution",
        "i_v_grid_distribution", "e_v_grid_distribution",
        "i_E_grid_distribution", "e_E_grid_distribution"
        ]
        self.selected_plot = tk.StringVar(value=self.plot_types[0])

        ttk.Label(root, text="Select Plot Type:").pack(pady=5)
        self.dropdown = ttk.Combobox(root, textvariable=self.selected_plot, values=self.plot_types, state="readonly")
        self.dropdown.pack(pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", self.update_plot)

        ttk.Button(root, text='Open File', command=self.read_file).pack(pady=5)
        
        self.z_idx = tk.IntVar(value=0)
        self.r_idx = tk.IntVar(value=0)

        ttk.Label(root, text="Z Index:").pack(pady=5)
        self.z_slider = ttk.Scale(root, from_=0, to=m-1, variable=self.z_idx, orient=tk.HORIZONTAL, command=self.update_plot)
        self.z_slider.pack(pady=5, fill=tk.X)

        ttk.Label(root, text="R Index:").pack(pady=5)
        self.r_slider = ttk.Scale(root, from_=0, to=n-1, variable=self.r_idx, orient=tk.HORIZONTAL, command=self.update_plot)
        self.r_slider.pack(pady=5, fill=tk.X)

        # Matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        #self.fig.set_facecolor('#24292e')

        self.get_data()
        # Initial plot
        self.update_plot()


    def read_file(self):
        file_path = filedialog.askopenfilename(title="Select Parameter File", filetypes=[("HDF5 Files", "*.h5*")])
        self.dump_file = file_path
        self.get_data
        self.update_plot()

    def get_data(self):      
        self.particle_data_i = read_particles_as_matrix(self.dump_file, self.ions_name)
        self.particle_data_e = read_particles_as_matrix(self.dump_file, 'electrons')
        #print(self.particle_data_e.shape)
        self.cells_i = get_cell_indicies(self.particle_data_i, self.z, self.r, self.m, self.n)
        self.cells_e = get_cell_indicies(self.particle_data_e, self.z, self.r, self.m, self.n)
        self.E_i, self.E_dist_i = get_energy_distribution(self.particle_data_i, mp, self.bins)
        self.E_e, self.E_dist_e = get_energy_distribution(self.particle_data_e, me, self.bins)
        self.V_i, self.V_dist_i = get_velocity_distribution(self.particle_data_i, self.bins)
        self.V_e, self.V_dist_e = get_velocity_distribution(self.particle_data_e, self.bins)
        self.z_e, self.z_dist_e = get_z_distribution(self.particle_data_e, self.z, self.bins)
        self.z_i, self.z_dist_i = get_z_distribution(self.particle_data_i, self.z, self.bins)
        self.r_e, self.r_dist_e = get_r_distribution(self.particle_data_e, self.r, self.bins)
        self.r_i, self.r_dist_i = get_r_distribution(self.particle_data_i, self.r, self.bins)

        Z = np.linspace(0, self.z, self.m-1)
        R = np.linspace(0, self.r, self.n-1)

        self.Z, self.R = np.meshgrid(Z, R)

        self.Ni = get_numerical_density(self.cells_i, self.z, self.r, self.m, self.n, self.np2c)
        self.Ne = get_numerical_density(self.cells_e, self.z, self.r, self.m, self.n, self.np2c)
        self.rho_i = get_plasma_density(self.Ni, -qe)
        self.rho_e = get_plasma_density(self.Ne, qe)
        self.Ti = get_temperature_distribution(self.particle_data_i, self.cells_i, self.m, self.n, mp)
        self.Te = get_temperature_distribution(self.particle_data_e, self.cells_e, self.m, self.n, me)
        self.dl = get_debye_length(self.Ne, self.Ni, self.Te, self.Ti)

        self.bins_v_i, self.V_dist_spacial_i = get_distribution_function_V(self.particle_data_i, self.cells_i, self.m, self.n, self.bins)
        self.bins_v_e, self.V_dist_spacial_e = get_distribution_function_V(self.particle_data_e, self.cells_e, self.m, self.n, self.bins)
        self.bins_E_i, self.E_dist_spacial_i = get_distribution_function_E(self.particle_data_i, self.cells_i, mp, self.m, self.n, self.bins)
        self.bins_E_e, self.E_dist_spacial_e = get_distribution_function_E(self.particle_data_e, self.cells_e, me, self.m, self.n, self.bins)


    def update_plot(self, event=None):
        """Update the plot based on dropdown selection"""
        plot_type = self.selected_plot.get()
        
        # Clear previous figure
        self.fig.clf()
        
        # Create new axis
        self.ax = self.fig.add_subplot(111)

        Z = np.linspace(0, self.z, self.m-1)
        R = np.linspace(0, self.r, self.n-1)
        ZZ, RR = np.meshgrid(Z, R)
        
        # For heatmap plots
        if plot_type.endswith("_heatmap"):
            # Create a meshgrid for plotting
            
            
            if plot_type == "Te_heatmap":
                im = self.ax.pcolormesh(Z, R, self.Te.T, cmap="plasma", shading='auto')
                self.ax.set_title("Electron Temperature")
                self.fig.colorbar(im, ax=self.ax, label="Temperature (eV)")
                
            elif plot_type == "Ti_heatmap":
                im = self.ax.pcolormesh(Z, R, self.Ti.T, cmap="plasma", shading='auto')
                self.ax.set_title("Ion Temperature")
                self.fig.colorbar(im, ax=self.ax, label="Temperature (eV)")
                
            elif plot_type == "ne_heatmap":
                im = self.ax.pcolormesh(Z, R, self.Ne.T, cmap="plasma", shading='auto')
                self.ax.set_title("Electron Number Density")
                self.fig.colorbar(im, ax=self.ax, label="Particles/m³")
                
            elif plot_type == "ni_heatmap":
                im = self.ax.pcolormesh(Z, R, self.Ni.T, cmap="plasma", shading='auto')
                self.ax.set_title("Ion Number Density")
                self.fig.colorbar(im, ax=self.ax, label="Particles/m³")
                
            elif plot_type == "rhoe_heatmap":
                im = self.ax.pcolormesh(Z, R, self.rho_e.T, cmap="plasma", shading='auto')
                self.ax.set_title("Electron Charge Density")
                self.fig.colorbar(im, ax=self.ax, label="C/m³")
                
            elif plot_type == "rhoi_heatmap":
                im = self.ax.pcolormesh(Z, R, self.rho_i.T, cmap="plasma", shading='auto')
                self.ax.set_title("Ion Charge Density")
                self.fig.colorbar(im, ax=self.ax, label="C/m³")

            elif plot_type == "debye_heatmap":
                im = self.ax.pcolormesh(Z, R, self.dl.T, cmap="plasma", shading='auto')
                self.ax.set_title("Debye Length")
                self.fig.colorbar(im, ax=self.ax, label="m")


            # Set common labels for heatmap plots
            self.ax.set_xlabel("Z (m)")
            self.ax.set_ylabel("R (m)")
            self.ax.set_aspect('equal')


        elif plot_type.endswith("_surface"):
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            parameter_name = plot_type.split("_")[0]
        
            
            if parameter_name == "Te":
                surf = self.ax.plot_surface(ZZ, RR, np.log10(self.Te.T+0.1), cmap="plasma", 
                                        edgecolor='none', alpha=0.8)
                self.ax.set_title("Electron Temperature log10")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="Temperature (eV)")
                #print('min T', np.min(self.Te))
                
            elif parameter_name == "Ti":
                surf = self.ax.plot_surface(ZZ, RR, self.Ti.T, cmap="plasma", 
                                        edgecolor='none', alpha=0.8)
                self.ax.set_title("Ion Temperature")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="Temperature (eV)")
                
            elif parameter_name == "ne":
                surf = self.ax.plot_surface(ZZ, RR, self.Ne.T, cmap="plasma", 
                                        edgecolor='none', alpha=0.8)
                self.ax.set_title("Electron Number Density")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="Particles/m³")
                
            elif parameter_name == "ni":
                surf = self.ax.plot_surface(ZZ, RR, self.Ni.T, cmap="plasma", 
                                        edgecolor='none', alpha=0.8)
                self.ax.set_title("Ion Number Density")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="Particles/m³")
                
            elif parameter_name == "rhoe":
                # For charge density, we need to be careful with the colormap
                norm = plt.Normalize(vmin=-np.max(abs(self.rho_e)), 
                                    vmax=np.max(abs(self.rho_e)))
                surf = self.ax.plot_surface(ZZ, RR, self.rho_e.T, cmap="plasma")
                self.ax.set_title("Electron Charge Density")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="C/m³")
                
            elif parameter_name == "rhoi":
                norm = plt.Normalize(vmin=-np.max(abs(self.rho_i)), 
                                    vmax=np.max(abs(self.rho_i)))
                surf = self.ax.plot_surface(ZZ, RR, self.rho_i.T, cmap="plasma")
                self.ax.set_title("Ion Charge Density")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="C/m³")
            
            elif parameter_name == "debye":
                norm = plt.Normalize(vmin=-np.max(abs(self.rho_i)), 
                                    vmax=np.max(abs(self.rho_i)))
                surf = self.ax.plot_surface(ZZ, RR, self.dl.T, cmap="plasma")
                self.ax.set_title("Debye length")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="m")
            
            # Set common labels for surface plots
            self.ax.set_xlabel("Z (m)")
            self.ax.set_ylabel("R (m)")
            self.ax.set_zlabel(parameter_name)
            
            # Set better viewing angle
            self.ax.view_init(elev=30, azim=45)


        # For distribution plots
        elif plot_type.endswith("_distribution"):

            i, j = self.z_idx.get(), self.r_idx.get()

            if plot_type == "e_velocity_distribution":
                self.ax.plot(self.V_e, self.V_dist_e, 'b-', linewidth=2)
                self.ax.set_title("Electron Velocity Distribution")
                self.ax.set_xlabel("Velocity (m/s)")
                self.ax.set_ylabel("Probability Density")
                
                
                # Add Maxwell-Boltzmann distribution for comparison if enough particles
                if self.particle_data_e.shape[1] > 10 and self.boltzman:
                    T_e = get_plasma_temperature(self.particle_data_e, me) * T_ratio
                    v = np.linspace(0, np.max(self.V_e), 100)
                    maxwell = 4*np.pi * (me/(2*np.pi*kb+j*T_e))**(3/2) * v**2 * np.exp(-me*v**2/(2*kb_j*T_e))
                    self.ax.plot(v, maxwell, 'r--', linewidth=1.5, label="Maxwell-Boltzmann")
                    self.ax.legend()
                

            elif plot_type == "i_velocity_distribution":
                self.ax.plot(self.V_i, self.V_dist_i, 'g-', linewidth=2)
                self.ax.set_title("Ion Velocity Distribution")
                self.ax.set_xlabel("Velocity (m/s)")
                self.ax.set_ylabel("Probability Density")
                
                
                # Add Maxwell-Boltzmann distribution for comparison if enough particles
                if self.particle_data_i.shape[1] > 10 and self.boltzman:
                    T_i = get_plasma_temperature(self.particle_data_i, mp) * T_ratio
                    v = np.linspace(0, np.max(self.V_i), 100)
                    maxwell = 4*np.pi * (mp/(2*np.pi*kb_j*T_i))**(3/2) * v**2 * np.exp(-mp*v**2/(2*kb_j*T_i))
                    self.ax.plot(v, maxwell, 'r--', linewidth=1.5, label="Maxwell-Boltzmann")
                    self.ax.legend()
                
            
            elif plot_type == "e_energy_distribution":
                self.ax.plot(self.E_e, self.E_dist_e, 'b-', linewidth=2)
                self.ax.set_title("Electron Energy Distribution")
                self.ax.set_xlabel("Energy (eV)")
                self.ax.set_ylabel("Probability Density")
                
                
                # Add Maxwellian energy distribution for comparison if enough particles
                if self.particle_data_e.shape[1] > 10 and self.boltzman:
                    T_e = get_plasma_temperature(self.particle_data_e, me) * T_ratio
                    E = np.linspace(0, np.max(self.E_e), 100)
                    maxwellian = 2 * np.pi * (1/(np.pi*kb*T_e))**(3/2) * np.sqrt(E) * np.exp(-E/(kb*T_e))
                    self.ax.plot(E, maxwellian, 'r--', linewidth=1.5, label="Maxwellian")
                    self.ax.legend()
                
            
            elif plot_type == "i_energy_distribution":
                self.ax.plot(self.E_i, self.E_dist_i, 'g-', linewidth=2)
                self.ax.set_title("Ion Energy Distribution")
                self.ax.set_xlabel("Energy (eV)")
                self.ax.set_ylabel("Probability Density")
                
                
                # Add Maxwellian energy distribution for comparison if enough particles
                if self.particle_data_i.shape[1] > 10 and self.boltzman:
                    T_i = get_plasma_temperature(self.particle_data_i, mp) * T_ratio
                    E = np.linspace(0, np.max(self.E_i), 100)
                    maxwellian = 2 * np.pi * (1/(np.pi*kb*T_i))**(3/2) * np.sqrt(E) * np.exp(-E/(kb*T_i))
                    self.ax.plot(E, maxwellian, 'r--', linewidth=1.5, label="Maxwellian")
                    self.ax.legend()
                
            elif plot_type == "zi_distribution":
                self.ax.plot(self.z_i, self.z_dist_i, 'g-', linewidth=2)
                self.ax.set_title("Ion Z Distribution")
                self.ax.set_xlabel("z (m)")
                self.ax.set_ylabel("Probability Density")
            
            elif plot_type == "ze_distribution":
                self.ax.plot(self.z_e, self.z_dist_e, 'g-', linewidth=2)
                self.ax.set_title("Electron Z Distribution")
                self.ax.set_xlabel("z (m)")
                self.ax.set_ylabel("Probability Density")
            
            elif plot_type == "ri_distribution":
                self.ax.plot(self.r_i, self.r_dist_i, 'g-', linewidth=2)
                self.ax.set_title("Ion R Distribution")
                self.ax.set_xlabel("r (m)")
                self.ax.set_ylabel("Probability Density")
            
            elif plot_type == "re_distribution":
                self.ax.plot(self.r_e, self.r_dist_e, 'g-', linewidth=2)
                self.ax.set_title("Electron R Distribution")
                self.ax.set_xlabel("r (m)")
                self.ax.set_ylabel("Probability Density")
            
            elif plot_type == "i_v_grid_distribution":
                self.ax.plot(self.bins_v_i, self.V_dist_spacial_i[i, j], label=f"Cell ({i}, {j})")
                self.ax.set_xlabel("Velocity Magnitude")
                self.ax.set_ylabel("Probability Density")
                self.ax.set_title("Velocity Distribution")

            elif plot_type == "e_v_grid_distribution":
                self.ax.plot(self.bins_v_e, self.V_dist_spacial_e[i, j], label=f"Cell ({i}, {j})")
                self.ax.set_xlabel("Energy")
                self.ax.set_ylabel("Probability Density")
                self.ax.set_title("Energy Distribution")
            
            elif plot_type == "i_E_grid_distribution":
                self.ax.plot(self.bins_E_i, self.E_dist_spacial_i[i, j], label=f"Cell ({i}, {j})")
                self.ax.set_xlabel("Velocity Magnitude")
                self.ax.set_ylabel("Probability Density")
                self.ax.set_title("Velocity Distribution")

            elif plot_type == "e_E_grid_distribution":
                self.ax.plot(self.bins_E_e, self.E_dist_spacial_e[i, j], label=f"Cell ({i}, {j})")
                self.ax.set_xlabel("Energy")
                self.ax.set_ylabel("Probability Density")
                self.ax.set_title("Energy Distribution")
            
        
        # Redraw canvas
        self.canvas.draw()

if __name__ == '__main__':

    bins = 128
    path = 'dump/Ar_magnetron_20mA_270V.h5'
    i_name = 'Arions'
    m, n, Z, R = read_grid_dimension(path)
    np2c = read_np2c(path, i_name)
    
    root = tk.Tk()
    app = PlotApp(root, i_name, path, m, n, Z, R, bins, boltzman=True, np2c = np2c)
    root.mainloop()
    

    #parts = read_particles_as_matrix(path, 'ions')
    #cells = get_cell_indicies(parts, Z, R, m, n)
    #r, v_dist = get_distribution_function_V(parts, cells, m, n, bins)

    #plot_distribution(r, v_dist, 3, 5)