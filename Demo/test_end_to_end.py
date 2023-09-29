# this file runs end to end latency eval
# to run: EMS-Pipeline/Demo $pytest test_end_to_end.py
from pytestqt.qt_compat import qt_api
from GUI import MainWindow


# def test_main_window(qtbot):
#     """
#     Basic santiy check to ensure we are setting up a QApplication
#     properly and are able to display the main GUI window
#     """
#     assert qt_api.QtWidgets.QApplication.instance() is not None
#     Window = MainWindow(1000,1000) #random size window
#     Window.show()
#     qtbot.addWidget(Window)
#     assert Window.isVisible()
#     assert Window.windowTitle() == 'CognitiveEMS Demo'

def test_run(qtbot):
    # GUI: Create the main window, show it, and run the app
    Window = MainWindow(1000,1000) #random size window
    Window.show()
    qtbot.addWidget(Window)

    # select whisper button
    Window.WhisperRadioButton.click()

    # select hayden recording
    Window.ComboBox.setCurrentText("000_190105")

    # push start button
    try:
        Window.StartButton.click()
    except Exception as e:
        print(e)

    def check_label():
        assert Window.windowTitle() == 'keep going' 
    
    qtbot.waitUntil(check_label)



