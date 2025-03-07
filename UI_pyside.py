from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QGridLayout, QFrame, QMenuBar
from PySide6.QtCore import Qt
import sys
import plotly.graph_objects as go
from PySide6.QtWebEngineWidgets import QWebEngineView

class ProcessFrame(QFrame):
    def __init__(self, process_id, parent=None):
        super().__init__(parent)
        self.process_id = process_id
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setStyleSheet("background-color: #3e3e3e; color: white;")

        layout = QVBoxLayout()
        self.info_label_voltage = QLabel("Voltage: 0")
        self.info_label_pressure = QLabel("Pressure: 0")
        layout.addWidget(self.info_label_voltage)
        layout.addWidget(self.info_label_pressure)

        self.plot_view = QWebEngineView()
        layout.addWidget(self.plot_view)
        self.setLayout(layout)
        self.update_plot()

    def update_plot(self, time_steps=None, particle_counts=None, target_currents=None):
        if time_steps is None:
            import numpy as np
            time_steps = np.linspace(0, 20e-6, 100)
            particle_counts = np.sin(10 * np.pi * time_steps)
            target_currents = np.cos(10 * np.pi * time_steps)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_steps, y=particle_counts, mode='lines', name='Particle Count', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=time_steps, y=target_currents, mode='lines', name='Target Current', line=dict(color='red'), yaxis='y2'))
        
        fig.update_layout(
            xaxis_title='Simulation Time',
            yaxis=dict(title='Particle Count', titlefont=dict(color='blue')),
            yaxis2=dict(title='Target Current', titlefont=dict(color='red'), overlaying='y', side='right'),
            template='plotly_dark'
        )
        
        self.plot_view.setHtml(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    def update_info(self, voltage, pressure):
        self.info_label_voltage.setText(f"Voltage: {voltage}")
        self.info_label_pressure.setText(f"Pressure: {pressure}")


class DashboardApp(QMainWindow):
    def __init__(self, num_processes=3):
        super().__init__()
        self.setWindowTitle("PIC Simulation Dashboard")
        self.setGeometry(100, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)
        
        self.process_frames = []
        for i in range(num_processes):
            row, col = divmod(i, 4)
            frame = ProcessFrame(i)
            self.layout.addWidget(frame, row, col)
            self.process_frames.append(frame)
        
        self.create_menu()

    def create_menu(self):
        menu_bar = self.menuBar()
        tools_menu = menu_bar.addMenu("Tools")
        tools_menu.addAction("Settings")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = DashboardApp(8)
    main_window.show()
    sys.exit(app.exec())
