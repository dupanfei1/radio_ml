# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/lab548/Desktop/93611/top/tabwidget126.ui'
#
# Created by: PyQt5 UI code generator 5.10.dev1801181815
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(945, 769)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralWidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 931, 731))
        self.tabWidget.setMinimumSize(QtCore.QSize(900, 0))
        self.tabWidget.setMaximumSize(QtCore.QSize(931, 16777215))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.tabWidget.setFont(font)
        self.tabWidget.setIconSize(QtCore.QSize(30, 29))
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.pushButton_17 = QtWidgets.QPushButton(self.tab)
        self.pushButton_17.setGeometry(QtCore.QRect(360, 230, 151, 41))
        self.pushButton_17.setObjectName("pushButton_17")
        self.textEdit_7 = QtWidgets.QTextEdit(self.tab)
        self.textEdit_7.setGeometry(QtCore.QRect(360, 80, 171, 31))
        self.textEdit_7.setObjectName("textEdit_7")
        self.label_20 = QtWidgets.QLabel(self.tab)
        self.label_20.setGeometry(QtCore.QRect(100, 80, 261, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.pushButton_6 = QtWidgets.QPushButton(self.tab)
        self.pushButton_6.setGeometry(QtCore.QRect(360, 310, 91, 41))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_21 = QtWidgets.QLabel(self.tab)
        self.label_21.setGeometry(QtCore.QRect(540, 90, 67, 17))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.pushButton_18 = QtWidgets.QPushButton(self.tab)
        self.pushButton_18.setGeometry(QtCore.QRect(360, 160, 151, 41))
        self.pushButton_18.setObjectName("pushButton_18")
        self.tabWidget.addTab(self.tab, "")
        self.transmit = QtWidgets.QWidget()
        self.transmit.setObjectName("transmit")
        self.label_10 = QtWidgets.QLabel(self.transmit)
        self.label_10.setGeometry(QtCore.QRect(240, 190, 41, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.textEdit = QtWidgets.QTextEdit(self.transmit)
        self.textEdit.setGeometry(QtCore.QRect(390, 310, 171, 71))
        self.textEdit.setObjectName("textEdit")
        self.label_6 = QtWidgets.QLabel(self.transmit)
        self.label_6.setGeometry(QtCore.QRect(190, 130, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.pushButton_2 = QtWidgets.QPushButton(self.transmit)
        self.pushButton_2.setGeometry(QtCore.QRect(490, 400, 71, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.textEdit_2 = QtWidgets.QTextEdit(self.transmit)
        self.textEdit_2.setGeometry(QtCore.QRect(390, 130, 171, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_7 = QtWidgets.QLabel(self.transmit)
        self.label_7.setGeometry(QtCore.QRect(210, 250, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_3 = QtWidgets.QLabel(self.transmit)
        self.label_3.setGeometry(QtCore.QRect(570, 200, 67, 17))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.comboBox_4 = QtWidgets.QComboBox(self.transmit)
        self.comboBox_4.setGeometry(QtCore.QRect(390, 250, 171, 31))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.label_4 = QtWidgets.QLabel(self.transmit)
        self.label_4.setGeometry(QtCore.QRect(230, 320, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.textEdit_3 = QtWidgets.QTextEdit(self.transmit)
        self.textEdit_3.setGeometry(QtCore.QRect(390, 190, 171, 31))
        self.textEdit_3.setObjectName("textEdit_3")
        self.label_2 = QtWidgets.QLabel(self.transmit)
        self.label_2.setGeometry(QtCore.QRect(570, 140, 67, 17))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.transmit)
        self.pushButton.setGeometry(QtCore.QRect(390, 400, 91, 31))
        self.pushButton.setObjectName("pushButton")
        self.tabWidget.addTab(self.transmit, "")
        self.receive = QtWidgets.QWidget()
        self.receive.setObjectName("receive")
        self.label_31 = QtWidgets.QLabel(self.receive)
        self.label_31.setGeometry(QtCore.QRect(200, 140, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_31.setFont(font)
        self.label_31.setObjectName("label_31")
        self.label_9 = QtWidgets.QLabel(self.receive)
        self.label_9.setGeometry(QtCore.QRect(200, 300, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_8 = QtWidgets.QLabel(self.receive)
        self.label_8.setGeometry(QtCore.QRect(200, 240, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.textBrowser = QtWidgets.QTextBrowser(self.receive)
        self.textBrowser.setGeometry(QtCore.QRect(390, 240, 171, 31))
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_8 = QtWidgets.QPushButton(self.receive)
        self.pushButton_8.setGeometry(QtCore.QRect(480, 370, 81, 31))
        self.pushButton_8.setObjectName("pushButton_8")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.receive)
        self.textBrowser_2.setGeometry(QtCore.QRect(390, 290, 171, 71))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.label_32 = QtWidgets.QLabel(self.receive)
        self.label_32.setGeometry(QtCore.QRect(200, 190, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_32.setFont(font)
        self.label_32.setObjectName("label_32")
        self.pushButton_7 = QtWidgets.QPushButton(self.receive)
        self.pushButton_7.setGeometry(QtCore.QRect(390, 370, 81, 31))
        self.pushButton_7.setObjectName("pushButton_7")
        self.textBrowser_8 = QtWidgets.QTextBrowser(self.receive)
        self.textBrowser_8.setGeometry(QtCore.QRect(390, 190, 171, 31))
        self.textBrowser_8.setObjectName("textBrowser_8")
        self.textBrowser_7 = QtWidgets.QTextBrowser(self.receive)
        self.textBrowser_7.setGeometry(QtCore.QRect(390, 140, 171, 31))
        self.textBrowser_7.setObjectName("textBrowser_7")
        self.label_5 = QtWidgets.QLabel(self.receive)
        self.label_5.setGeometry(QtCore.QRect(570, 150, 67, 17))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.pushButton_9 = QtWidgets.QPushButton(self.receive)
        self.pushButton_9.setGeometry(QtCore.QRect(420, 90, 111, 31))
        self.pushButton_9.setObjectName("pushButton_9")
        self.tabWidget.addTab(self.receive, "")
        self.perform = QtWidgets.QWidget()
        self.perform.setObjectName("perform")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.perform)
        self.graphicsView_3.setGeometry(QtCore.QRect(10, 160, 441, 451))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.graphicsView_4 = QtWidgets.QGraphicsView(self.perform)
        self.graphicsView_4.setGeometry(QtCore.QRect(470, 160, 441, 451))
        brush = QtGui.QBrush(QtGui.QColor(22, 0, 0))
        brush.setStyle(QtCore.Qt.NoBrush)
        self.graphicsView_4.setBackgroundBrush(brush)
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.label_35 = QtWidgets.QLabel(self.perform)
        self.label_35.setGeometry(QtCore.QRect(140, 640, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_35.setFont(font)
        self.label_35.setObjectName("label_35")
        self.label_36 = QtWidgets.QLabel(self.perform)
        self.label_36.setGeometry(QtCore.QRect(550, 640, 281, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_36.setFont(font)
        self.label_36.setObjectName("label_36")
        self.pushButton_3 = QtWidgets.QPushButton(self.perform)
        self.pushButton_3.setGeometry(QtCore.QRect(110, 106, 221, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.perform)
        self.pushButton_4.setGeometry(QtCore.QRect(600, 106, 201, 41))
        self.pushButton_4.setObjectName("pushButton_4")
        self.tabWidget.addTab(self.perform, "")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 945, 31))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.LeftToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)


        self.textEdit_3.setText("10")
        self.textEdit_2.setText("1000.0000")
        self.textEdit.setText("10100001")

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.pushButton_8.clicked.connect(MainWindow.stop_rece)
        self.pushButton.clicked.connect(MainWindow.begin_trans)
        self.pushButton_2.clicked.connect(MainWindow.stop_trans)
        self.pushButton_7.clicked.connect(MainWindow.begin_rece)
        self.pushButton_4.clicked.connect(MainWindow.showsnrs)
        self.pushButton_3.clicked.connect(MainWindow.showconfus)
        self.pushButton_17.clicked.connect(MainWindow.reset9361)
        self.pushButton_6.clicked.connect(MainWindow.exit0)
        self.pushButton_18.clicked.connect(MainWindow.configuratefreq)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_17.setText(_translate("MainWindow", "Reset AD9361"))
        self.label_20.setText(_translate("MainWindow", "Central frequency (Receiver)"))
        self.pushButton_6.setText(_translate("MainWindow", "Exit"))
        self.label_21.setText(_translate("MainWindow", "MHz"))
        self.pushButton_18.setText(_translate("MainWindow", "Configurate"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Configuration"))
        self.label_10.setText(_translate("MainWindow", "SNR"))
        self.label_6.setText(_translate("MainWindow", "Central frequency"))
        self.pushButton_2.setText(_translate("MainWindow", "Stop"))
        self.label_7.setText(_translate("MainWindow", "Modulate mode"))
        self.label_3.setText(_translate("MainWindow", "dB"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "BPSK"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "QPSK"))
        self.label_4.setText(_translate("MainWindow", "Input bits"))
        self.label_2.setText(_translate("MainWindow", "MHz"))
        self.pushButton.setText(_translate("MainWindow", "Transmit"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.transmit), _translate("MainWindow", "Transmit"))
        self.label_31.setText(_translate("MainWindow", "Central Frequency"))
        self.label_9.setText(_translate("MainWindow", "Demodulate bits"))
        self.label_8.setText(_translate("MainWindow", "Recognition mode"))
        self.pushButton_8.setText(_translate("MainWindow", "Stop"))
        self.label_32.setText(_translate("MainWindow", "SampleFreq"))
        self.pushButton_7.setText(_translate("MainWindow", "Update"))
        self.label_5.setText(_translate("MainWindow", "MHz"))
        self.pushButton_9.setText(_translate("MainWindow", "Receive"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.receive), _translate("MainWindow", "Receive"))
        self.label_35.setText(_translate("MainWindow", "Confusion Matrix"))
        self.label_36.setText(_translate("MainWindow", "Classification Accuracy for SNRs"))
        self.pushButton_3.setText(_translate("MainWindow", "show confuse-matrix"))
        self.pushButton_4.setText(_translate("MainWindow", "show accuracy-curve"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.perform), _translate("MainWindow", "Performance(off-line)"))

