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

    def __init__(self, parent=None):
        super(DataReceive, self).__init__(parent)
        self.tcpServer = QTcpServer(self)
        if self.tcpServer.listen(QHostAddress("127.0.0.1"), 8890):
            self.tcpServer.newConnection.connect(self.sendMessage)
            self.tcpServerConnection = None
            print('init done')

        self.index = 0
        self.data_size = 0
        self.message = None
        self.image = None

    def sendMessage(self):
        print('send')
        self.data_size = 0
        self.tcpServerConnection = self.tcpServer.nextPendingConnection()
        self.tcpServerConnection.readyRead.connect(self.readData)

    def readData(self):
        in_data = QDataStream(self.client)
        in_data.setVersion(QDataStream.Qt_4_0)
        if self.data_size == 0:
            tmp = self.client.bytesAvailable()
            if tmp < SIZEOF_UINT32:
                return
            self.data_size = in_data.readUInt32()
            print(self.data_size)
        if self.client.bytesAvailable() < self.data_size:
            return

        self.message = self.client.read(self.data_size)

        self.list.append(self.message)
        print(self.index)
        self.index = self.index + 1

        if self.index == 10:
            try:
                with open("message.pkl", "wb") as fp:
                    pickle.dump(self.list, fp, pickle.HIGHEST_PROTOCOL)
                self.client.close()
            except:
                print("what")

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

import pickle

class Client(QObject):
    Signal_receiveDone = pyqtSignal()

    def __init__(self, parent=None):
        super(Client, self).__init__(parent)
        self.client = QTcpSocket(self)
        self.client.abort()
        self.client.connectToHost("127.0.0.1", 8890)
        self.client.readyRead.connect(self.readData)

        self.data_size = 0
        self.message = None
        self.image = None
        self.list = []
    # def sendMessage(self):
    #     print('send')
    #     self.data_size = 0
    #     self.tcpServerConnection = self.tcpServer.nextPendingConnection()
    #     self.tcpServerConnection.readyRead.connect(self.readData)
        self.index = 0
    def readData(self):
        self.message = QByteArray()
        in_data = QDataStream(self.client)
        in_data.setVersion(QDataStream.Qt_4_0)
        if self.data_size == 0:
            tmp = self.client.bytesAvailable()
            if tmp < SIZEOF_UINT32:
                return
            self.data_size = in_data.readUInt32()
            print(self.data_size)
        if self.client.bytesAvailable() < self.data_size:
            return

        self.message = self.client.read(self.data_size)

        self.list.append(self.message)
        print(self.index)
        self.index = self.index + 1

        if self.index == 10:
            try:
                with open("message.pkl", "wb") as fp:
                    pickle.dump(self.list, fp, pickle.HIGHEST_PROTOCOL)
                self.client.close()
            except:
                print("what")

    def parse_message(self, message):
        print(message)


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.layout = QGridLayout()
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)

        self.dataReceive = DataReceive(self)
        # self.client = Client(self)


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


