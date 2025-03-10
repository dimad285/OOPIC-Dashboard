import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ProcessData
import shutil
import numpy as np

NUM_PROCESSES = 3  # Default number of simulated processes
COLS = 4  # Number of columns in the grid layout

# Define color themes
LIGHT_THEME = {"bg": "#f0f0f0", "fg": "#000000", "frame_bg": "#ffffff", "frame_fg": "#000000"}
DARK_THEME = {"bg": "#2e2e2e", "fg": "#ffffff", "frame_bg": "#3e3e3e", "frame_fg": "#ffffff"}
CURRENT_THEME = LIGHT_THEME  # Change to LIGHT_THEME for a light mode
INPUT_SYMBOLS = {'voltage': 'V', 'pressure': 'P'}
    



class ProcessFrame:
    def __init__(self, parent, font_size:int, process_id):
        self.process_id = process_id
        self.frame = ttk.LabelFrame(parent, text=f"Process {process_id+1}", padding=10,)
        self.frame.configure(style="TFrame")
        
        self.table = ttk.Treeview(self.frame, columns=("Parameter", "Value", "Relative Value"), show="headings", height=6)
        self.table.heading("Parameter", text="Parameter")
        self.table.heading("Value", text="Value")
        self.table.heading("Relative Value", text="Relative Value")
        self.table.column("Parameter", width=120)
        self.table.column("Value", width=100)
        self.table.pack(pady=5, fill=tk.X)

        # Table default values
        self.parameters = [
            ("Min Debye", "0"),
            ("Larmor Radius", "0 Pa"),
            ("Plasma f", "0"),
            ("Cyclotron f", "0 A"),
            ("Collision f", "0 /m³"),
            ("Flyby Time", "0 K")
        ]
        for param, value in self.parameters:
            self.table.insert("", "end", values=(param, value))
        
        self.fig, self.ax1 = plt.subplots(figsize=(6, 4), tight_layout=True)
        self.ax2 = self.ax1.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig.patch.set_facecolor(CURRENT_THEME['frame_bg'])  # Dark gray figure background
        self.ax1.set_facecolor(CURRENT_THEME["bg"])  # Darker gray plot background
        self.ax2.set_facecolor(CURRENT_THEME["bg"])  # Darker gray plot background

        
    def update_table(self, process: ProcessData.ProcessData):
        """
        Updates the table values using data from the given ProcessData instance.

        Parameters:
        - process: ProcessData instance containing simulation data.
        """
        # Extract latest values from process data
        debye = f"{process.min_dl_e:.2f} m"
        lr = f"{process.min_larmor_radius:.2e} m"
        plf = f"{process.max_plasma_f:.2e} Hz"
        clf = f"{process.max_cycl_f:.2e} Hz"
        colf = f"{process.coll_f:.2e} Hz"
        flbt = f"{process.min_cell_fly_time:.2e} s"

        # New values for the table
        new_values = [
            ("Min Debye", debye),
            ("Larmor Radius", lr),
            ("Plasma f", plf),
            ("Cyclotron f", clf),
            ("Collision f", colf),
            ("Flyby Time", flbt)
        ]

        # Update table entries
        for i, (param, value) in enumerate(new_values):
            self.table.item(self.table.get_children()[i], values=(param, value))
    
    def update_plot(self, process: ProcessData.ProcessData):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(process.time_steps, process.particle_counts, label="Particle Count", color='blue')
        self.ax2.plot(process.time_steps, process.target_currents, label="Target Current", color='red')

        self.ax1.set_xlim(0, 20e-6)
        self.ax1.set_xlabel("Simulation Time", color=CURRENT_THEME["fg"])
        self.ax1.set_ylabel("Particle Count", color=CURRENT_THEME["fg"])
        self.ax1.legend(loc="upper left")
        #self.ax2.set_ylabel("Target Current", color='red', labelpad=15)
        self.ax2.legend(loc="upper right")
        self.ax1.tick_params(axis='y', labelcolor=CURRENT_THEME["fg"])
        self.ax2.tick_params(axis='y', labelcolor=CURRENT_THEME["fg"])
        self.ax1.tick_params(axis='x', labelcolor=CURRENT_THEME["fg"])
        self.ax2.spines['bottom'].set_color('white')
        self.ax2.spines['top'].set_color('white')
        self.ax2.spines['right'].set_color('white')
        self.ax2.spines['left'].set_color('white')

        self.fig.tight_layout()  # Ensure labels fit without overlapping
        self.canvas.draw()

    
    def grid(self, row, col):
        self.frame.grid(row=row, column=col, padx=10, pady=5, sticky="nsew")


class TitleScreen:
    def __init__(self, root, start_callback):
        self.root = root
        self.root.title("PIC Simulation Setup")
        self.root.configure(bg=CURRENT_THEME["bg"])

        ttk.Label(root, text="Number of Processes:", style="TLabel").pack()
        self.num_processes_entry = ttk.Entry(root)
        self.num_processes_entry.insert(0, str(NUM_PROCESSES))
        self.num_processes_entry.pack(pady=15)

        self.process_settings_frame = ttk.Frame(root)
        self.process_settings_frame.pack()

        self.set_process_count_button = ttk.Button(root, text="Set Processes", command=self.create_process_fields)
        self.set_process_count_button.pack()

        # Extra features
        

        self.process_settings = []
        self.extra_param_labels = []  # Store extra column labels
        self.extra_param_entries = []  # Store extra entry widgets
        self.start_callback = start_callback

        #ttk.Button(root, text="Read from File", command=self.load_from_file).pack()
        

    def create_process_fields(self):
        """Creates entry fields for each process."""
        for widget in self.process_settings_frame.winfo_children():
            widget.destroy()

        self.process_settings = []
        num_processes = int(self.num_processes_entry.get())

        # Header row
        ttk.Label(self.process_settings_frame, text="Start Voltage").grid(row=0, column=0, pady=5, padx=5)
        self.start_voltage_entry = ttk.Entry(self.process_settings_frame)
        self.start_voltage_entry.grid(row=0, column=1)

        ttk.Label(self.process_settings_frame, text="Step Voltage").grid(row=1, column=0, pady=5, padx=5)
        self.step_voltage_entry = ttk.Entry(self.process_settings_frame)
        self.step_voltage_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self.process_settings_frame, text="Pressure").grid(row=1, column=4)

        # Input rows
        for i in range(num_processes):
            voltage_entry = ttk.Entry(self.process_settings_frame)
            pressure_entry = ttk.Entry(self.process_settings_frame)
            process_label = ttk.Label(self.process_settings_frame, text=f"Process {i+1}")

            process_label.grid(row=i+2, column=0, pady=5, padx=5)
            voltage_entry.grid(row=i+2, column=1, pady=5, padx=5)
            pressure_entry.grid(row=i+2, column=4, pady=5, padx=5)


            self.process_settings.append((voltage_entry, pressure_entry))

        self.set_process_count_button.destroy()

        ttk.Button(root, text="Add Parameter Column", command=self.add_param_column).pack(pady=5)
        ttk.Button(root, text="Autofill Parameters", command=self.autofill_parameters).pack(pady=5)
        ttk.Button(root, text="Start", command=self.start_dashboard).pack(pady=5)

    def apply_voltage_step(self):
        """Auto-fills voltage values based on start and step voltage."""
        try:
            start_voltage = float(self.start_voltage_entry.get())
            step_voltage = float(self.step_voltage_entry.get())
            for i, (voltage_entry, _) in enumerate(self.process_settings):
                voltage_entry.delete(0, tk.END)
                voltage_entry.insert(0, str(start_voltage + i * step_voltage))
        except ValueError:
            pass  # Ignore errors if entries are empty

    def add_param_column(self):
        """Dynamically adds new parameter columns."""
        col_index = len(self.extra_param_labels) + 5  # Offset by existing columns

        # Create a frame for the label + rename button
        param_frame = ttk.Frame(self.process_settings_frame)
        param_frame.grid(row=1, column=col_index, padx=5)

        label = ttk.Label(param_frame, text=f"Param {col_index-4}")
        label.pack(side=tk.LEFT)

        rename_button = ttk.Button(param_frame, text="✎", command=lambda: self.rename_param(label))
        rename_button.pack(side=tk.LEFT)

        self.extra_param_labels.append(label)

        new_entries = []
        for i in range(len(self.process_settings)):
            entry = ttk.Entry(self.process_settings_frame)
            entry.grid(row=i+2, column=col_index)
            new_entries.append(entry)

        self.extra_param_entries.append(new_entries)

    def rename_param(self, label):
        """Allows renaming of extra parameter columns."""
        new_name = tk.simpledialog.askstring("Rename Parameter", "Enter new parameter name:")
        if new_name:
            label.config(text=new_name)

    def autofill_parameters(self):
        """Copies first process parameters to all others."""
        if self.process_settings:
            first_voltage = self.process_settings[0][0].get()
            first_pressure = self.process_settings[0][1].get()

            for voltage_entry, pressure_entry in self.process_settings[1:]:
                voltage_entry.delete(0, tk.END)
                voltage_entry.insert(0, first_voltage)
                pressure_entry.delete(0, tk.END)
                pressure_entry.insert(0, first_pressure)

            # Autofill extra columns
            for col_entries in self.extra_param_entries:
                first_value = col_entries[0].get()
                for entry in col_entries[1:]:
                    entry.delete(0, tk.END)
                    entry.insert(0, first_value)

        # Ensure voltage steps are applied correctly
        self.apply_voltage_step()

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
        
        
        self.create_input_files()
        self.processes = [ProcessData.ProcessData(i, f"input/input_{i}.inp", f"dump/1.h5") for i in range(NUM_PROCESSES)]
        for i, (voltage, pressure) in enumerate(voltage_pressure_data):
            self.processes[i].voltage = voltage
            self.processes[i].pressure = pressure
        
        self.frames = []
        self.create_frames()
        self.init_processes()
        self.update_data()
    
    def create_frames(self):
        for i in range(NUM_PROCESSES):
            row, col = divmod(i, COLS)
            frame = ProcessFrame(self.root, 12, i)
            frame.grid(row, col)
            self.frames.append(frame)
        
        for i in range(COLS):
            self.root.columnconfigure(i, weight=1)

    def init_processes(self):
        for process in self.processes:
            process.init_process()
    
    def create_input_files(self):
        for i in range(NUM_PROCESSES):
            shutil.copyfile("input/input.inp", f"input/input_{i}.inp")

    def update_data(self):
        if not self.root.winfo_exists():  # Check if window still exists
            return
        for i, process in enumerate(self.processes):
            if process.process_finished:
                self.frames[i].update_plot(process)
                self.frames[i].update_table(process)
                process.process_finished = False
                process.ready_for_launch = True
            elif process.ready_for_launch:
                process.start_process()
        
        self.root.after(1000, self.update_data)

if __name__ == "__main__":
    
    root = tk.Tk()
    style = ttk.Style()
    #style.configure("Process.TFrame", background=CURRENT_THEME["frame_bg"], foreground=CURRENT_THEME["frame_fg"])
    #style.configure("TLabelFrame", background=CURRENT_THEME["frame_bg"], foreground=CURRENT_THEME["frame_fg"])
    style.configure("TFrame", background=CURRENT_THEME["bg"])
    style.configure("TEntry", background=CURRENT_THEME["frame_bg"])
    style.configure("TButton", background=CURRENT_THEME["bg"])
    style.configure("TLabel", background=CURRENT_THEME["bg"], foreground=CURRENT_THEME["frame_fg"])
    root.option_add('*tearOff', False)
    menubar = tk.Menu(root)
    menubar.configure(bg=CURRENT_THEME["bg"], fg=CURRENT_THEME["fg"])
    menu_tools = tk.Menu(menubar)
    menubar.add_cascade(label='Tools', menu=menu_tools)
    menu_tools.add_command(label='Settings')
    root['menu'] = menubar
    TitleScreen(root, lambda data: DashboardApp(tk.Tk(), data))
    root.mainloop()

