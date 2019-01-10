import sys

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QObject
from PyQt5.QtCore import QByteArray
from PyQt5.QtCore import QDataStream
from PyQt5.QtNetwork import QTcpServer
from PyQt5.Qt import QApplication
from PyQt5.Qt import QHostAddress
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLabel

SIZEOF_UINT64 = 8
SIZEOF_UINT32 = 4

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtNetwork import *

class DataReceive(QObject):
    Signal_receiveDone = pyqtSignal()

    def __init__(self, parent=None):
        super(DataReceive, self).__init__(parent)
        self.tcpServer = QTcpServer()
        if not self.tcpServer.listen(QHostAddress.Any, 8888):
            print('port 10001 is busy, please change the port!')
            self.close()
        self.tcpServer.newConnection.connect(self.sendMessage)
        self.tcpServerConnection = None
        self.data_size = 0
        self.message = None
        self.image = None

    def sendMessage(self):
        self.data_size = 0
        self.tcpServerConnection = self.tcpServer.nextPendingConnection()
        self.tcpServerConnection.readyRead.connect(self.readData)

    def readData(self):
        self.message = QByteArray()
        in_data = QDataStream(self.tcpServerConnection)
        in_data.setVersion(QDataStream.Qt_4_0)
        if self.data_size == 0:
            tmp = self.tcpServerConnection.bytesAvailable()
            if tmp < SIZEOF_UINT32:
                return
            self.data_size = in_data.readInt32()
        if self.tcpServerConnection.bytesAvailable() < self.data_size - 14:
            return

        self.message = self.tcpServerConnection.read(self.data_size - 14)
        self.parse_message(self.message)

    def parse_message(self, message):
        print(message)


    # def readImage(self, ba):
    #     img = QImage()
    #     # ss = QString.fromLatin1(ba, len(ba))
    #     # rc = QByteArray.fromBase64(ss.toLatin1())
    #     # rdc = qUncompress(rc)
    #     rdc = ba
    #     if len(rdc) < 1:
    #         print('bug')
    #         return
    #     print('ok')
    #     img.loadFromData(rdc)
    #     self.image = img
    #     self.Signal_receiveDone.emit()
    #
    # def getImage(self):
    #     return self.image


# class ProcessThread(QThread):
#     Signal_processDone = pyqtSignal()
#
#     def __init__(self):
#         super(ProcessThread, self).__init__()
#         self._isProcess = False
#         self._imageBuff = None
#         self._imageIn = None
#         self._imageOut = None
#
#     def run(self):
#         while True:
#             if self._isProcess:
#                 self.reset_isProcess()
#                 self._imageIn = self._imageBuff.copy()
#
#
#
#                 self.Signal_processDone.emit()
#                 print('processing3....')
#             self.msleep(3)
#
#     def set_isProcess(self):
#         self._isProcess = True
#
#     def reset_isProcess(self):
#         self._isProcess = False
#
#     def getImage(self):
#         return self._imageOut
#
#     def inputImage(self, image):
#         self._imageBuff = image


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.layout = QGridLayout()
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)

        self.dataReceive = DataReceive(self)


    def __del__(self):
        # if self.thread.isRunning():
        #    self.thread.exit(0)
        pass


def main():
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


