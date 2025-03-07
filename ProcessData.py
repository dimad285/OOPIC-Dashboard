import h5py
import subprocess
import threading
import os
import numpy as np
import parser

class ProcessData:
    def __init__(self, process_id, input_file_path, dump_file_path):
        self.process_id = process_id
        self.input_file_path = input_file_path
        self.dump_file_path = dump_file_path
        self.process = None  # Store the running process
        self.process_finished = False  # Track process completion
        self.ready_for_launch = True  # Track if process is ready to start
        self.time_steps = []
        self.particle_counts = []
        self.target_currents = []
        self.voltage = 0.0
        self.pressure = 0.0
        self.additional_info = "Initializing..."
        self.last_modified = 0
        self.particle_data = np.empty((5, 0))
        self.mp = 1.6726219e-27  # Mass of hydrogen
        self.qe = 1.60217662e-19  # Elementary charge
        self.me = 9.10938356e-31  # Mass of electron
        self.A = 1.0  # Atomic number of hydrogen
        self.kb = 1.38064852e-23  # Boltzmann constant
        self.e0 = 8.85418782e-12  # Vacuum permittivity

        self.m = 0
        self.n = 0
        self.z = 0
        self.r = 0

        self.test_command = 'C:/Users/Dima/anaconda3/python.exe test_launch.py'

    def cmdStartH5(self):
        cmd = f'oopicpro -i {self.input_file_path}.inp -nox -h5 -od {self.dump_file_path}' 
        return cmd
    
    def cmdH5(self, cycl, thread):
        cmd = f'oopicpro -i {self.input_file_path}{thread}.inp -nox -s {cycl} -h5 -or -d dump/bin/{thread} -sf dump/h5/{thread} -dp {cycl}' 
        return cmd
    
    def init_process(self):
        os.system(self.cmdStartH5())
        read_list = ['J', 'K', 'Z', 'R', 'A']
        input_params = parser.read_parameters(self.input_file_path, read_list)
        if 'J' in input_params:
            self.m = input_params['J']
        if 'K' in input_params:
            self.n = input_params['K']
        if 'Z' in input_params:
            self.z = input_params['Z']
        if 'R' in input_params:
            self.r = input_params['R']
        if 'A' in input_params:
            self.A = input_params['A']

    def start_process(self):
        """Launch the simulation process in a separate thread."""
        if not self.ready_for_launch:
            print(f"Process {self.process_id} is not ready to launch.")
            return
        
        self.process_finished = False
        self.ready_for_launch = False  # Mark as running
        
        def run_simulation():
            print(f"Starting process {self.process_id}...")
            try:
                self.process = subprocess.Popen(self.test_command, shell=True)
            except:
                print(f"Error starting process {self.process_id}.")
                self.process_finished = True
                return
            print(f"Process {self.process_id} started.")
            self.process.wait()  # Block until process finishes
            self.update_from_file()  # Update data after process finishes
            self.process_finished = True  # Mark process as completed
            print(f"Process {self.process_id} finished.")
        
            #self.ready_for_launch = True  # Mark as ready for new launch

        thread = threading.Thread(target=run_simulation)
        thread.start()
    
    def update_from_file(self):
        """Check if the HDF5 file has new data and update attributes."""
        if not os.path.exists(self.dump_file_path):
            return False  # File does not exist, no update needed
        
        if self.process and self.process.poll() is None:
            return False  # Process is still running, wait before updating
        
        modified_time = os.path.getmtime(self.dump_file_path)
        #if modified_time == self.last_modified:
        #    return False  # No changes detected, skip update
        
        self.last_modified = modified_time
        self.read_particles_as_matrix("ions")
        particle_count = self.particle_data.shape[1]
        target_current = self.update_target_current(threshold=1)
        
        self.time_steps.append(len(self.time_steps)*1e-6)
        self.particle_counts.append(particle_count)
        self.target_currents.append(target_current)
        print(f"Process {self.process_id} updated with {particle_count} particles and {target_current} current.")
        
        self.additional_info = f"Particles: {particle_count}, Current: {target_current}, Voltage: {self.voltage}, Pressure: {self.pressure}"
        return True  # Update happened
    
    def read_particles_as_matrix(self, particle_name: str):
        try:
            with h5py.File(self.dump_file_path, 'r') as f:
                if particle_name not in f:
                    return np.empty((5, 0))
                
                data_list = []
                for group_name in f[particle_name].keys():
                    group = f[particle_name][group_name]
                    if 'ptcls' in group:
                        data = group['ptcls'][()]
                        if data.shape[1] == 5:
                            data_list.append(data)
                
                if data_list:
                    self.particle_data = np.vstack(data_list).T
                else:
                    self.particle_data = np.empty((5, 0))
        except (OSError, KeyError, ValueError):
            self.particle_data = np.empty((5, 0))
    
    def update_target_current(self, threshold: float = 1e-3):
        mask = self.particle_data[0] < threshold
        return np.sum(self.particle_data[3][mask]) * self.qe / threshold


    def update_species_properties(self):
        # Implement logic to update species properties
        return 0.0
    
    def update_temperature(self):
        v = self.particle_data[2:, :]
        e_mean = np.mean(np.sum(v**2, axis=0)) * self.mp * self.A
        return 2 * e_mean / (3 * self.kb)
    
    def compute_spatial_distribution(self):
        """Compute local temperature and density on a 2D grid"""
        x, y = self.particle_data[:2]  # Extract positions
        vx, vy, vz = self.particle_data[2:]

        # Define grid
        x_bins = np.linspace(np.min(x), np.max(x), self.m + 1)
        y_bins = np.linspace(np.min(y), np.max(y), self.n + 1)
        
        local_temperature = np.zeros((self.m, self.n))
        local_density = np.zeros((self.m, self.n))
        particle_counts = np.zeros((self.m, self.n))

        # Assign particles to grid cells
        x_indices = np.digitize(x, x_bins) - 1
        y_indices = np.digitize(y, y_bins) - 1
        
        for i in range(self.particle_data.shape[1]):
            xi, yi = x_indices[i], y_indices[i]
            if 0 <= xi < self.m and 0 <= yi < self.n:
                kinetic_energy = 0.5 * self.mp * self.A * (vx[i]**2 + vy[i]**2 + vz[i]**2)
                local_temperature[xi, yi] += kinetic_energy
                local_density[xi, yi] += 1
                particle_counts[xi, yi] += 1
        
        # Normalize
        local_temperature /= (3 * self.kb * np.maximum(particle_counts, 1))
        local_density /= self.x * self.y / (self.m * self.n)
        local_debye_length = np.sqrt(self.e0 * self.kb * local_temperature / (local_density * self.qe**2))
        
        return local_temperature, local_density, local_debye_length
    
    def read_input_file(self, param_list):
        # Implement logic to read input file
        return None
    
    def change_input_file(self, param_list):
        # Implement logic to change input file
        return None