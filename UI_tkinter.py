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
        for i, process in enumerate(self.processes):
            if process.finished:
                self.frames[i].update_plot(process.time_steps, process.particle_counts, process.target_currents, process.additional_info)
                process.finished = False
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
