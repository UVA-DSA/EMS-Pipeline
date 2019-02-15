import sys
from PyQt4.QtCore import pyqtSignal, QSize, Qt
from PyQt4.QtGui import *

class MyWidget(QWidget):

    clicked = pyqtSignal()
    keyPressed = pyqtSignal(unicode)
    
    def __init__(self, parent = None):
    
        QWidget.__init__(self, parent)
        self.color = QColor(0, 0, 0)
    
    def paintEvent(self, event):
    
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), QBrush(self.color))
        painter.end()
    
    def keyPressEvent(self, event):
    
        self.keyPressed.emit(event.text())
        event.accept()
    
    def mousePressEvent(self, event):
    
        self.setFocus(Qt.OtherFocusReason)
        event.accept()
    
    def mouseReleaseEvent(self, event):
    
        if event.button() == Qt.LeftButton:
        
            self.color = QColor(self.color.green(), self.color.blue(),
                                127 - self.color.red())
            self.update()
            self.clicked.emit()
            event.accept()
    
    def sizeHint(self):
    
        return QSize(100, 100)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = QWidget()
    
    mywidget = MyWidget()
    label = QLabel()
    
    mywidget.clicked.connect(label.clear)
    mywidget.keyPressed.connect(label.setText)
    
    layout = QHBoxLayout()
    layout.addWidget(mywidget)
    layout.addWidget(label)
    window.setLayout(layout)
    
    window.show()
    sys.exit(app.exec_())

'''
import sys
from PyQt4.QtCore import pyqtSlot
from PyQt4.QtGui import *

# create our window
app = QApplication(sys.argv)

w = QWidget()
w.setGeometry(500, 100, 1000, 600)
w.setWindowTitle('CognitiveEMS Demo')
# Set window size.
#w.resize(320, 150)

# Create textbox for Speech
SpeechBox = QLineEdit(w)
SpeechBox.move(20, 20)
SpeechBox.resize(280, 200)

# Create textbox for NLP
NLPBox = QLineEdit(w)
NLPBox.move(300, 20)
NLPBox.resize(280, 200)
 

 
# Create a button in the window
button = QPushButton('Click me', w)
button.move(20, 300)
 
# Create the actions
@pyqtSlot()
def on_click():
    SpeechBox.setText("Button clicked.")
 
# connect the signals to the slots
button.clicked.connect(on_click)
 
# Show the window and run the app
w.show()
app.exec_()
'''

