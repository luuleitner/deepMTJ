"""
	#deepMTJ
	an open-source software tool made for biomechanical researchers

	Copyright (C) 2020 by the authors: Jarolim Robert (University of Graz), <robert.jarolim@uni-graz.at> and
	Leitner Christoph (Graz University of Technology), <christoph.leitner@tugraz.at>.
	
	Please note: main.ui was compiled in open-source Qt (protected by GPLv3)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
