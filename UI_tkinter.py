import tkinter as tk
from tkinter import ttk, filedialog
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess
import threading
import os
import numpy as np

NUM_PROCESSES = 3  # Default number of simulated processes
COLS = 4  # Number of columns in the grid layout

# Define color themes
LIGHT_THEME = {"bg": "#f0f0f0", "fg": "#000000", "frame_bg": "#ffffff", "frame_fg": "#000000"}
DARK_THEME = {"bg": "#2e2e2e", "fg": "#ffffff", "frame_bg": "#3e3e3e", "frame_fg": "#ffffff"}
CURRENT_THEME = DARK_THEME  # Change to LIGHT_THEME for a light mode

    
import subprocess
import threading
import os
import h5py
import numpy as np

class ProcessData:
    def __init__(self, process_id, file_path, command):
        self.process_id = process_id
        self.file_path = file_path
        self.command = command  # Command to run the simulation
        self.process = None  # Store the running process
        self.time_steps = []
        self.particle_counts = []
        self.target_currents = []
        self.voltage = 0.0
        self.pressure = 0.0
        self.additional_info = "Initializing..."
        self.last_modified = 0
        self.particle_data = np.empty((5, 0))
    
    def start_process(self):
        """Launch the simulation process in a separate thread."""
        def run_simulation():
            self.process = subprocess.Popen(self.command, shell=True)
            self.process.wait()  # Block until process finishes
            print(f"Process {self.process_id} finished.")

        thread = threading.Thread(target=run_simulation)
        thread.start()
    
    def update_from_file(self):
        """Check if the HDF5 file has new data and update attributes."""
        if not os.path.exists(self.file_path):
            return False  # File does not exist, no update needed
        
        if self.process and self.process.poll() is None:
            return False  # Process is still running, wait before updating
        
        modified_time = os.path.getmtime(self.file_path)
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
            with h5py.File(self.file_path, 'r') as f:
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

    
    def update_temperature(self):
        # Implement logic to calculate temperature
        return 0.0
    
    def update_density(self):
        # Implement logic to calculate density
        return 0.0
    
    def update_debye_length(self):
        # Implement logic to calculate Debye length
        return 0.0


class ProcessFrame:
    def __init__(self, parent, process_id):
        self.process_id = process_id
        self.frame = ttk.LabelFrame(parent, text=f"Process {process_id+1}", padding=10)
        self.frame.configure(style="Process.TFrame")
        
        self.info_label = ttk.Label(self.frame, text="Initializing...", background=CURRENT_THEME["frame_bg"], foreground=CURRENT_THEME["frame_fg"])
        self.info_label.pack()
        
        self.fig, self.ax1 = plt.subplots(figsize=(6, 4))
        self.ax2 = self.ax1.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame)
        self.canvas.get_tk_widget().pack()
    
    def update_plot(self, time_steps, particle_counts, target_currents, info_text):
        self.ax1.clear()
        self.ax1.plot(time_steps, particle_counts, label="Particle Count", color='blue')
        self.ax1.set_xlabel("Simulation Time")
        self.ax1.set_ylabel("Particle Count", color='blue')
        self.ax1.legend()

        self.ax1.set_xbound(0, 20e-6)
        
        self.ax2.clear()
        self.ax2.plot(time_steps, target_currents, label="Target Current", color='red')
        self.ax2.set_ylabel("Target Current", color='red')
        self.ax2.legend()
        
        self.canvas.draw()
        self.info_label.config(text=info_text)
    
    def grid(self, row, col):
        self.frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")


class TitleScreen:
    def __init__(self, root, start_callback):
        self.root = root
        self.root.title("PIC Simulation Setup")
        
        ttk.Label(root, text="Number of Processes:").pack()
        self.num_processes_entry = ttk.Entry(root)
        self.num_processes_entry.insert(0, str(NUM_PROCESSES))
        self.num_processes_entry.pack()
        
        self.process_settings_frame = ttk.Frame(root)
        self.process_settings_frame.pack()
        
        ttk.Button(root, text="Set Processes", command=self.create_process_fields).pack()
        
        self.process_settings = []
        self.start_callback = start_callback
        
        ttk.Button(root, text="Read from File", command=self.load_from_file).pack()
        ttk.Button(root, text="Start", command=self.start_dashboard).pack()
    
    def create_process_fields(self):
        for widget in self.process_settings_frame.winfo_children():
            widget.destroy()
        
        self.process_settings = []
        num_processes = int(self.num_processes_entry.get())
        
        for i in range(num_processes):
            frame = ttk.Frame(self.process_settings_frame)
            ttk.Label(frame, text=f"Process {i+1} Voltage:").pack(side=tk.LEFT)
            voltage_entry = ttk.Entry(frame)
            voltage_entry.pack(side=tk.LEFT)
            ttk.Label(frame, text="Pressure:").pack(side=tk.LEFT)
            pressure_entry = ttk.Entry(frame)
            pressure_entry.pack(side=tk.LEFT)
            frame.pack()
            self.process_settings.append((voltage_entry, pressure_entry))
    
    def load_from_file(self):
        file_path = filedialog.askopenfilename(title="Select Parameter File", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= len(self.process_settings):
                    for i, (voltage_entry, pressure_entry) in enumerate(self.process_settings):
                        parts = lines[i].strip().split()
                        if len(parts) >= 2:
                            voltage_entry.delete(0, tk.END)
                            voltage_entry.insert(0, parts[0])
                            pressure_entry.delete(0, tk.END)
                            pressure_entry.insert(0, parts[1])
    
    def start_dashboard(self):
        global NUM_PROCESSES
        NUM_PROCESSES = int(self.num_processes_entry.get())
        
        voltage_pressure_data = [(float(v.get()), float(p.get())) for v, p in self.process_settings]
        
        self.root.destroy()
        self.start_callback(voltage_pressure_data)


class DashboardApp:
    def __init__(self, root, voltage_pressure_data):
        self.root = root
        self.root.title("PIC Simulation Dashboard")
        self.root.configure(bg=CURRENT_THEME["bg"])
        
        style = ttk.Style()
        style.configure("Process.TFrame", background=CURRENT_THEME["frame_bg"], foreground=CURRENT_THEME["frame_fg"])
        
        self.processes = [ProcessData(i, f"dump/1.h5") for i in range(NUM_PROCESSES)]
        for i, (voltage, pressure) in enumerate(voltage_pressure_data):
            self.processes[i].voltage = voltage
            self.processes[i].pressure = pressure
        
        self.frames = []
        self.create_frames()
        self.update_data()
    
    def create_frames(self):
        for i in range(NUM_PROCESSES):
            row, col = divmod(i, COLS)
            frame = ProcessFrame(self.root, i)
            frame.grid(row, col)
            self.frames.append(frame)
        
        for i in range(COLS):
            self.root.columnconfigure(i, weight=1)
    
    def update_data(self):
        if not self.root.winfo_exists():  # Check if window still exists
            return
        updated = False
        for i, process in enumerate(self.processes):
            if process.update_from_file():
                self.frames[i].update_plot(process.time_steps, process.particle_counts, process.target_currents, process.additional_info)
                updated = True
        
        self.root.after(500 if updated else 2000, self.update_data)

if __name__ == "__main__":
    root = tk.Tk()
    root.option_add('*tearOff', False)
    menubar = tk.Menu(root)
    menu_tools = tk.Menu(menubar)
    menubar.add_cascade(label='Tools', menu=menu_tools)
    menu_tools.add_command(label='Settings')
    root['menu'] = menubar
    TitleScreen(root, lambda data: DashboardApp(tk.Tk(), data))
    root.mainloop()
