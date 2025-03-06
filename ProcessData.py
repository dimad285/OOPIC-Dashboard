import h5py
import subprocess
import threading
import os
import numpy as np

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

    def cmdStartH5(self, thread):
        cmd = f'oopicpro -i {self.input_file_path}{thread}.inp -nox -h5 -od {self.dump_file_path}/{thread}' 
        return cmd
    
    def cmdH5(self, cycl, thread):
        cmd = f'oopicpro -i {self.input_file_path}{thread}.inp -nox -s {cycl} -h5 -or -d dump/bin/{thread} -sf dump/h5/{thread} -dp {cycl}' 
        return cmd
    

    def start_process(self):
        """Launch the simulation process in a separate thread."""
        if not self.ready_for_launch:
            print(f"Process {self.process_id} is not ready to launch.")
            return
        
        self.process_finished = False
        self.ready_for_launch = False  # Mark as running
        
        def run_simulation():
            self.process = subprocess.Popen(self.command, shell=True)
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
        if modified_time == self.last_modified:
            return False  # No changes detected, skip update
        
        self.last_modified = modified_time
        self.read_particles_as_matrix("ions")
        particle_count = self.particle_data.shape[1]
        target_current = self.update_target_current()
        
        self.time_steps.append(len(self.time_steps))
        self.particle_counts.append(particle_count)
        self.target_currents.append(target_current)
        
        self.additional_info = f"Particles: {particle_count}, Current: {target_current}, Voltage: {self.voltage}, Pressure: {self.pressure}"
        return True  # Update happened
    
    def read_particles_as_matrix(self, particle_name: str):
        try:
            with h5py.File(self.dump_file_path + '/1.h5', 'r') as f:
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
        return np.sum(self.particle_data[3][mask])


    def update_species_properties(self):
        # Implement logic to update species properties
        return 0.0
    
    def update_temperature(self):
        # Implement logic to calculate temperature
        return 0.0
    
    def update_density(self):
        # Implement logic to calculate density
        return 0.0
    
    def update_debye_length(self):
        # Implement logic to calculate Debye length
        return 0.0
    
    def read_input_file(self):
        # Implement logic to read input file
        return None
    
    def change_input_file(self):
        # Implement logic to change input file
        return None