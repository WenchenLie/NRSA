from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Analysis import SDOFmodel
import os
import sys
import time
import datetime
import multiprocessing
from typing import Literal
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog, QWidget
from ui.Win import Ui_Win

class Win(QDialog):

    def __init__(self, main: SDOFmodel) -> None:
        super().__init__()
        self.ui = Ui_Win()
        self.ui.setupUi(self)
        self.main = main
        self.init_ui()

    def init_ui(self):
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint)
