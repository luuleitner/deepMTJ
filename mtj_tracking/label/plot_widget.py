# Imports
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure


# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)


# Matplotlib widget
class PlotWidget(QtWidgets.QWidget):
    coordinate_selection = pyqtSignal(object)

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.toolbar)
        self.setLayout(self.vbl)

        self.canvas.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def loadFrame(self, frame, coord):
        self.canvas.fig.clear()
        ax = self.canvas.figure.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(frame)
        if coord is not None:
            ax.scatter(coord[0], coord[1], color="red")
        self.canvas.draw()

    def onclick(self, event):
        ix, iy = event.xdata, event.ydata
        if ix is None or iy is None:
            return
        self.coordinate_selection.emit((np.round(ix), np.round(iy)))

    def clear(self):
        self.canvas.fig.clear()
        self.canvas.draw()