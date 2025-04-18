from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QGridLayout, QLabel, QSlider,
                             QVBoxLayout, QWidget)


class OpacityWidget(QWidget):
    def __init__(self, features: List[str], parent=None):
        super().__init__(parent)
        layout: QVBoxLayout = QGridLayout(self)
        self.widget_features = {}
        layout.addWidget(QLabel("Feature"), 0, 0)
        layout.addWidget(QLabel("Show"), 0, 1)
        layout.addWidget(QLabel("Opacity"), 0, 2)


        for index, feature in enumerate(features):
            label = QLabel(feature)
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setMinimumWidth(100)
            slider.setValue(100)

            bla = {"checkbox": checkbox, "slider": slider}
            self.widget_features[feature] = bla
            layout.addWidget(label, index+1, 0)
            layout.addWidget(checkbox, index+1, 1)
            layout.addWidget(slider, index+1, 2)
        
        self.setLayout(layout)