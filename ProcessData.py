import h5py
import subprocess
import threading
import os
import numpy as np
import parser
import time
import postprocessing
import diagnostics

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

        self.max_ne = 0
        self.max_ne_idx = [0, 0]
        self.max_ni = 0
        self.max_ni_idx = [0, 0]
        self.min_dl_e = 0
        self.min_larmor_radius = 0
        self.min_cell_fly_time = 0
        self.max_cycl_f = 0
        self.max_plasma_f = 0
        self.coll_f = 0

        self.m = 128
        self.n = 64
        self.z = 0.128
        self.r = 0.064

        self.test_command = 'C:/Users/Dima/anaconda3/python.exe test_launch.py'

        self.process_time = 0

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
            self.end_time = time.time()
        
            #self.ready_for_launch = True  # Mark as ready for new launch

        thread = threading.Thread(target=run_simulation)
        self.start_time = time.time()
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

        self.np2c = postprocessing.read_np2c(self.dump_file_path, 'electrons')
        self.part_idx = postprocessing.get_cell_indicies(self.particle_data, self.z, self.r, self.m, self.n)
        self.ne = postprocessing.get_numerical_density(self.part_idx, self.z, self.r, self.m, self.n, self.np2c)
        self.max_ne, self.max_ne_idx = diagnostics.max_value_in_array(self.ne)
        self.Te = postprocessing.get_temperature_distribution(self.particle_data, self.part_idx, self.m, self.n, self.me)
        self.Te_max = self.Te[self.max_ne_idx[0], self.max_ne_idx[1]]
        self.min_dl_e = diagnostics.debye_length(self.max_ne, self.Te_max)
        self.max_v = diagnostics.max_velocity(self.particle_data)
        self.min_larmor_radius = diagnostics.larmor_radius(self.max_v, 1, self.qe, self.me)
        self.min_cell_fly_time = np.hypot(self.z/(self.m-1), self.r/(self.n-1)) / self.max_v
        self.max_plasma_f = diagnostics.plasma_frequency(self.max_ne, self.qe, self.me)
        self.max_cycl_f = diagnostics.cyclotron_frequency(1, self.qe, self.me)
        self.coll_f = diagnostics.collision_frequency(1, self.max_v, 1)
        
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
    
    
    def read_input_file(self, param_list):
        # Implement logic to read input file
        return None
    
    def change_input_file(self, param_list):
        # Implement logic to change input file
        return None