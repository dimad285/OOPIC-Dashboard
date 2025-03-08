import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import h5py
import scipy as sp

kb = 1.38064852e-23  # Boltzmann constant
qe = -1.60217662e-19  # Elementary charge    
mp = 1.6726219e-27  # Mass of hydrogen
e0 = 8.85418782e-12  # Vacuum permittivity
me = 9.1e-31


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

    cell_indicies = np.zeros((2, particle_data.shape[1]), dtype=np.int32)

    if particle_data.shape[1] == 0:
        print('0 particles')
        return cell_indicies

    dz = Z / (m - 1)
    dr = R / (n - 1)

    # Normalize particle positions to grid
    x_norm = particle_data[0, :] / dz
    y_norm = particle_data[1, :] / dr

    # Find the cell index for each particle
    cell_indicies[0, :] = np.floor(x_norm).astype(int)
    cell_indicies[1, :] = np.floor(y_norm).astype(int) 

    return cell_indicies


def get_numerical_density(cell_indices, Z, R, m, n):
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

    return nd


def get_energy_distribution(particle_data, M, bins = 64):

    if particle_data.shape[1] == 0:
        print('0 particles')
        return (np.arange(bins), np.zeros(bins))
    
    v = particle_data[2:, :]
    E = (np.sum(v**2, axis=0)) * M * 0.5
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
    e_mean = np.mean(np.sum(v**2, axis=0)) * M
    return e_mean / (3 * kb)

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

    # Accumulate sum of squared speeds per cell
    np.add.at(T, (cell_indices[0], cell_indices[1]), v_squared)

    # Count the number of particles in each cell
    np.add.at(particle_count, (cell_indices[0], cell_indices[1]), 1)

    # Avoid division by zero
    mask = particle_count > 0
    T[mask] = (M / kb) * (T[mask] / particle_count[mask])

    return T


def get_plasma_density(numerical_density, q):
    return numerical_density * q

def get_debye_length(particle_data):

    pass

def distribution_function(particle_data, cell_indicies, X, Y, m, n):

    pass



class PlotApp:
    def __init__(self, root, file_path, m, n, z, r, bins, boltzman = False):
        self.root = root
        self.root.title("Plot Selector")
        self.dump_file = file_path
        self.m = m
        self.n = n
        self.z = z
        self.r = r
        self.bins = bins
        self.boltzman = boltzman

        # Dropdown menu
        self.plot_types = [
        "Te_heatmap", "Ti_heatmap", 
        "ne_heatmap", "ni_heatmap", 
        "rhoe_heatmap", "rhoi_heatmap", 
        "Te_surface", "Ti_surface", 
        "ne_surface", "ni_surface", 
        "rhoe_surface", "rhoi_surface",
        "e_velocity_distribution", "i_velocity_distribution",
        "e_energy_distribution", "i_energy_distribution"
        ]
        self.selected_plot = tk.StringVar(value=self.plot_types[0])

        ttk.Label(root, text="Select Plot Type:").pack(pady=5)
        self.dropdown = ttk.Combobox(root, textvariable=self.selected_plot, values=self.plot_types, state="readonly")
        self.dropdown.pack(pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", self.update_plot)

        # Matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.get_data()
        # Initial plot
        self.update_plot()

    def get_data(self):      
        self.particle_data_i = read_particles_as_matrix(self.dump_file, 'ions')
        self.particle_data_e = read_particles_as_matrix(self.dump_file, 'electrons')
        #print(self.particle_data_e.shape)
        self.cells_i = get_cell_indicies(self.particle_data_i, self.z, self.r, self.m, self.n)
        self.cells_e = get_cell_indicies(self.particle_data_e, self.z, self.r, self.m, self.n)
        self.E_i, self.E_dist_i = get_energy_distribution(self.particle_data_i, mp, self.bins)
        self.E_e, self.E_dist_e = get_energy_distribution(self.particle_data_e, me, self.bins)
        self.V_i, self.V_dist_i = get_velocity_distribution(self.particle_data_i, self.bins)
        self.V_e, self.V_dist_e = get_velocity_distribution(self.particle_data_e, self.bins)

        Z = np.linspace(0, self.z, self.m-1)
        R = np.linspace(0, self.r, self.n-1)

        self.Z, self.R = np.meshgrid(Z, R)

        self.Ni = get_numerical_density(self.cells_i, self.z, self.r, self.m, self.n)
        self.Ne = get_numerical_density(self.cells_e, self.z, self.r, self.m, self.n)
        self.rho_i = get_plasma_density(self.Ni, -qe)
        self.rho_e = get_plasma_density(self.Ne, qe)
        self.Ti = get_temperature_distribution(self.particle_data_i, self.cells_i, self.m, self.n, mp)
        self.Te = get_temperature_distribution(self.particle_data_e, self.cells_e, self.m, self.n, me)


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
                self.fig.colorbar(im, ax=self.ax, label="Temperature (K)")
                
            elif plot_type == "Ti_heatmap":
                im = self.ax.pcolormesh(Z, R, self.Ti.T, cmap="plasma", shading='auto')
                self.ax.set_title("Ion Temperature")
                self.fig.colorbar(im, ax=self.ax, label="Temperature (K)")
                
            elif plot_type == "ne_heatmap":
                im = self.ax.pcolormesh(Z, R, self.Ne.T, cmap="viridis", shading='auto')
                self.ax.set_title("Electron Number Density")
                self.fig.colorbar(im, ax=self.ax, label="Particles/m³")
                
            elif plot_type == "ni_heatmap":
                im = self.ax.pcolormesh(Z, R, self.Ni.T, cmap="viridis", shading='auto')
                self.ax.set_title("Ion Number Density")
                self.fig.colorbar(im, ax=self.ax, label="Particles/m³")
                
            elif plot_type == "rhoe_heatmap":
                im = self.ax.pcolormesh(Z, R, self.rho_e.T, cmap="RdBu", shading='auto', 
                                    norm=plt.Normalize(vmin=-np.max(abs(self.rho_e)), 
                                                        vmax=np.max(abs(self.rho_e))))
                self.ax.set_title("Electron Charge Density")
                self.fig.colorbar(im, ax=self.ax, label="C/m³")
                
            elif plot_type == "rhoi_heatmap":
                im = self.ax.pcolormesh(Z, R, self.rho_i.T, cmap="RdBu", shading='auto',
                                    norm=plt.Normalize(vmin=-np.max(abs(self.rho_i)), 
                                                        vmax=np.max(abs(self.rho_i))))
                self.ax.set_title("Ion Charge Density")
                self.fig.colorbar(im, ax=self.ax, label="C/m³")
            
            # Set common labels for heatmap plots
            self.ax.set_xlabel("Z (m)")
            self.ax.set_ylabel("R (m)")
            self.ax.set_aspect('equal')


        elif plot_type.endswith("_surface"):
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            parameter_name = plot_type.split("_")[0]

            
            if parameter_name == "Te":
                surf = self.ax.plot_surface(ZZ, RR, self.Te.T, cmap="plasma", 
                                        edgecolor='none', alpha=0.8)
                self.ax.set_title("Electron Temperature")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="Temperature (K)")
                
            elif parameter_name == "Ti":
                surf = self.ax.plot_surface(ZZ, RR, self.Ti.T, cmap="plasma", 
                                        edgecolor='none', alpha=0.8)
                self.ax.set_title("Ion Temperature")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="Temperature (K)")
                
            elif parameter_name == "ne":
                surf = self.ax.plot_surface(ZZ, RR, self.Ne.T, cmap="viridis", 
                                        edgecolor='none', alpha=0.8)
                self.ax.set_title("Electron Number Density")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="Particles/m³")
                
            elif parameter_name == "ni":
                surf = self.ax.plot_surface(ZZ, RR, self.Ni.T, cmap="viridis", 
                                        edgecolor='none', alpha=0.8)
                self.ax.set_title("Ion Number Density")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="Particles/m³")
                
            elif parameter_name == "rhoe":
                # For charge density, we need to be careful with the colormap
                norm = plt.Normalize(vmin=-np.max(abs(self.rho_e)), 
                                    vmax=np.max(abs(self.rho_e)))
                surf = self.ax.plot_surface(ZZ, RR, self.rho_e.T, cmap="RdBu", 
                                        norm=norm, edgecolor='none', alpha=0.8)
                self.ax.set_title("Electron Charge Density")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="C/m³")
                
            elif parameter_name == "rhoi":
                norm = plt.Normalize(vmin=-np.max(abs(self.rho_i)), 
                                    vmax=np.max(abs(self.rho_i)))
                surf = self.ax.plot_surface(ZZ, RR, self.rho_i.T, cmap="RdBu", 
                                        norm=norm, edgecolor='none', alpha=0.8)
                self.ax.set_title("Ion Charge Density")
                self.fig.colorbar(surf, ax=self.ax, pad=0.1, shrink=0.5, 
                                label="C/m³")
            
            # Set common labels for surface plots
            self.ax.set_xlabel("Z (m)")
            self.ax.set_ylabel("R (m)")
            self.ax.set_zlabel(parameter_name)
            
            # Set better viewing angle
            self.ax.view_init(elev=30, azim=45)


        # For distribution plots
        elif plot_type.endswith("_distribution"):
            if plot_type == "e_velocity_distribution":
                self.ax.plot(self.V_e, self.V_dist_e, 'b-', linewidth=2)
                self.ax.set_title("Electron Velocity Distribution")
                self.ax.set_xlabel("Velocity (m/s)")
                self.ax.set_ylabel("Probability Density")
                
                
                # Add Maxwell-Boltzmann distribution for comparison if enough particles
                if self.particle_data_e.shape[1] > 10 and self.boltzman:
                    T_e = get_plasma_temperature(self.particle_data_e, me)
                    v = np.linspace(0, np.max(self.V_e), 100)
                    maxwell = 4*np.pi * (me/(2*np.pi*kb*T_e))**(3/2) * v**2 * np.exp(-me*v**2/(2*kb*T_e))
                    self.ax.plot(v, maxwell, 'r--', linewidth=1.5, label="Maxwell-Boltzmann")
                    self.ax.legend()
                

            elif plot_type == "i_velocity_distribution":
                self.ax.plot(self.V_i, self.V_dist_i, 'g-', linewidth=2)
                self.ax.set_title("Ion Velocity Distribution")
                self.ax.set_xlabel("Velocity (m/s)")
                self.ax.set_ylabel("Probability Density")
                
                
                # Add Maxwell-Boltzmann distribution for comparison if enough particles
                if self.particle_data_i.shape[1] > 10 and self.boltzman:
                    T_i = get_plasma_temperature(self.particle_data_i, mp)
                    v = np.linspace(0, np.max(self.V_i), 100)
                    maxwell = 4*np.pi * (mp/(2*np.pi*kb*T_i))**(3/2) * v**2 * np.exp(-mp*v**2/(2*kb*T_i))
                    self.ax.plot(v, maxwell, 'r--', linewidth=1.5, label="Maxwell-Boltzmann")
                    self.ax.legend()
                
            
            elif plot_type == "e_energy_distribution":
                self.ax.plot(self.E_e, self.E_dist_e, 'b-', linewidth=2)
                self.ax.set_title("Electron Energy Distribution")
                self.ax.set_xlabel("Energy (J)")
                self.ax.set_ylabel("Probability Density")
                
                
                # Add Maxwellian energy distribution for comparison if enough particles
                if self.particle_data_e.shape[1] > 10 and self.boltzman:
                    T_e = get_plasma_temperature(self.particle_data_e, me)
                    E = np.linspace(0, np.max(self.E_e), 100)
                    maxwellian = 2 * np.pi * (1/(np.pi*kb*T_e))**(3/2) * np.sqrt(E) * np.exp(-E/(kb*T_e))
                    self.ax.plot(E, maxwellian, 'r--', linewidth=1.5, label="Maxwellian")
                    self.ax.legend()
                
            
            elif plot_type == "i_energy_distribution":
                self.ax.plot(self.E_i, self.E_dist_i, 'g-', linewidth=2)
                self.ax.set_title("Ion Energy Distribution")
                self.ax.set_xlabel("Energy (J)")
                self.ax.set_ylabel("Probability Density")
                
                
                # Add Maxwellian energy distribution for comparison if enough particles
                if self.particle_data_i.shape[1] > 10 and self.boltzman:
                    T_i = get_plasma_temperature(self.particle_data_i, mp)
                    E = np.linspace(0, np.max(self.E_i), 100)
                    maxwellian = 2 * np.pi * (1/(np.pi*kb*T_i))**(3/2) * np.sqrt(E) * np.exp(-E/(kb*T_i))
                    self.ax.plot(E, maxwellian, 'r--', linewidth=1.5, label="Maxwellian")
                    self.ax.legend()
                
        
        # Redraw canvas
        self.canvas.draw()

if __name__ == '__main__':

    bins = 512
    root = tk.Tk()
    app = PlotApp(root, 'dump/1.h5', 128, 64, 0.128, 0.064, bins, boltzman=True)
    root.mainloop()