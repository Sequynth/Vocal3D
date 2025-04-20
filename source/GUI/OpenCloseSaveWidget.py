from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QWidget


class OpenCloseSaveWidget(QWidget):
    fileOpenedSignal = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super(OpenCloseSaveWidget, self).__init__()
        self.setLayout(QHBoxLayout())

        self.addButton("Open", self.open)
        self.addButton("New Video", self.loadVideo)

        self.camera_calib_path: str = ""
        self.laser_calib_path: str = ""

    def addButton(self, title, function):
        button = QPushButton(title)
        button.clicked.connect(function)
        self.layout().addWidget(button)

    def open(self):
        self.camera_calib_path, _ = QFileDialog.getOpenFileName(self, 'Open Camera Calibration file', '', "Camera Calibration Files (*.json *.mat)")
        self.laser_calib_path, _ = QFileDialog.getOpenFileName(self, 'Open Laser Calibration file', '', "Laser Calibration Files (*.json *.mat)")
        self.video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', "Video Files (*.avi *.mp4 *.mkv *.AVI *.MP4)")
        self.fileOpenedSignal.emit(self.camera_calib_path, self.laser_calib_path, self.video_path)

    def loadVideo(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', "Video Files (*.avi *.mp4 *.mkv *.AVI *.MP4)")
        self.fileOpenedSignal.emit(self.camera_calib_path, self.laser_calib_path, self.video_path)
