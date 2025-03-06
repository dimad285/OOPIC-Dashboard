import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ProcessData

NUM_PROCESSES = 3  # Default number of simulated processes
COLS = 4  # Number of columns in the grid layout

# Define color themes
LIGHT_THEME = {"bg": "#f0f0f0", "fg": "#000000", "frame_bg": "#ffffff", "frame_fg": "#000000"}
DARK_THEME = {"bg": "#2e2e2e", "fg": "#ffffff", "frame_bg": "#3e3e3e", "frame_fg": "#ffffff"}
CURRENT_THEME = DARK_THEME  # Change to LIGHT_THEME for a light mode

    



class ProcessFrame:
    def __init__(self, parent, font_size:int, process_id):
        self.process_id = process_id
        self.frame = ttk.LabelFrame(parent, text=f"Process {process_id+1}", padding=10)
        self.frame.configure(style="Process.TFrame")
        
        self.info_label_voltage = ttk.Label(self.frame, text="voltage: 0", background=CURRENT_THEME["frame_bg"], foreground=CURRENT_THEME["frame_fg"],
                                            font=("Arial", font_size))
        self.info_label_pressure = ttk.Label(self.frame,  text="pressure: 0", background=CURRENT_THEME["frame_bg"], foreground=CURRENT_THEME["frame_fg"],
                                            font=("Arial", font_size))
        self.info_label_voltage.pack()
        self.info_label_pressure.pack()
    
        
        self.fig, self.ax1 = plt.subplots(figsize=(6, 4), tight_layout=True)
        self.ax2 = self.ax1.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig.patch.set_facecolor("#2e2e2e")  # Dark gray figure background
        self.ax1.set_facecolor("#3e3e3e")  # Darker gray plot background
        self.ax2.set_facecolor("#3e3e3e")

        
    
    def update_plot(self, process: ProcessData.ProcessData):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(process.time_steps, process.particle_counts, label="Particle Count", color='blue')
        self.ax2.plot(process.time_steps, process.target_currents, label="Target Current", color='red')

        self.ax1.set_xlim(0, 20e-6)
        self.ax1.set_xlabel("Simulation Time", color='white')
        self.ax1.set_ylabel("Particle Count", color='blue')
        self.ax1.legend(loc="upper left")
        #self.ax2.set_ylabel("Target Current", color='red', labelpad=15)
        self.ax2.legend(loc="upper right")
        self.ax1.tick_params(axis='y', labelcolor='white')
        self.ax2.tick_params(axis='y', labelcolor='white')
        self.ax1.tick_params(axis='x', labelcolor='white')
        self.ax1.spines['bottom'].set_color('white')
        self.ax1.spines['left'].set_color('white')
        self.ax2.spines['right'].set_color('white')
        
        self.fig.tight_layout()  # Ensure labels fit without overlapping
        self.canvas.draw()

        if self.info_label_voltage != None:
            self.info_label_voltage.config(text=f'voltage: {process.voltage}')
        if self.info_label_pressure != None:
            self.info_label_pressure.config(text=f'pressure: {process.pressure}')
    
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

        # Extra features
        ttk.Button(root, text="Add Parameter Column", command=self.add_param_column).pack()
        ttk.Button(root, text="Autofill Parameters", command=self.autofill_parameters).pack()

        self.process_settings = []
        self.extra_param_labels = []  # Store extra column labels
        self.extra_param_entries = []  # Store extra entry widgets
        self.start_callback = start_callback

        ttk.Button(root, text="Read from File", command=self.load_from_file).pack()
        ttk.Button(root, text="Start", command=self.start_dashboard).pack()

    def create_process_fields(self):
        """Creates entry fields for each process."""
        for widget in self.process_settings_frame.winfo_children():
            widget.destroy()

        self.process_settings = []
        num_processes = int(self.num_processes_entry.get())

        # Header row
        ttk.Label(self.process_settings_frame, text="Start Voltage").grid(row=0, column=0)
        self.start_voltage_entry = ttk.Entry(self.process_settings_frame)
        self.start_voltage_entry.grid(row=0, column=1)

        ttk.Label(self.process_settings_frame, text="Step Voltage").grid(row=0, column=2)
        self.step_voltage_entry = ttk.Entry(self.process_settings_frame)
        self.step_voltage_entry.grid(row=0, column=3)

        ttk.Label(self.process_settings_frame, text="Pressure").grid(row=0, column=4)

        # Input rows
        for i in range(num_processes):
            voltage_entry = ttk.Entry(self.process_settings_frame)
            pressure_entry = ttk.Entry(self.process_settings_frame)

            voltage_entry.grid(row=i+1, column=1)
            pressure_entry.grid(row=i+1, column=4)

            self.process_settings.append((voltage_entry, pressure_entry))

        self.apply_voltage_step()  # Auto-fill voltage

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
        param_frame.grid(row=0, column=col_index, padx=5)

        label = ttk.Label(param_frame, text=f"Param {col_index-4}")
        label.pack(side=tk.LEFT)

        rename_button = ttk.Button(param_frame, text="✎", command=lambda: self.rename_param(label))
        rename_button.pack(side=tk.LEFT)

        self.extra_param_labels.append(label)

        new_entries = []
        for i in range(len(self.process_settings)):
            entry = ttk.Entry(self.process_settings_frame)
            entry.grid(row=i+1, column=col_index)
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
        
        style = ttk.Style()
        style.configure("Process.TFrame", background=CURRENT_THEME["frame_bg"], foreground=CURRENT_THEME["frame_fg"])
        
        self.processes = [ProcessData.ProcessData(i, f"input/input.txt", f"dump/1.h5") for i in range(NUM_PROCESSES)]
        for i, (voltage, pressure) in enumerate(voltage_pressure_data):
            self.processes[i].voltage = voltage
            self.processes[i].pressure = pressure
        
        self.frames = []
        self.create_frames()
        #self.init_processes()
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
    
    def update_data(self):
        if not self.root.winfo_exists():  # Check if window still exists
            return
        for i, process in enumerate(self.processes):
            if process.process_finished:
                self.frames[i].update_plot(process)
                process.process_finished = False
                process.ready_for_launch = True
            elif process.ready_for_launch:
                process.start_process()
        
        self.root.after(1000, self.update_data)

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
