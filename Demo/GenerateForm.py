import commands
from Form_Filling import textParse2
from Form_Filling import prescription_form2
from classes import GUISignal
import subprocess
import datetime

def GenerateForm(Window, text):

    # Create Signal Objects
    MsgSignal = GUISignal()
    MsgSignal.signal.connect(Window.UpdateMsgBox)

    ButtonsSignal = GUISignal()
    ButtonsSignal.signal.connect(Window.ButtonsSetEnabled)

    ButtonsSignal.signal.emit([(Window.GenerateFormButton, False)])

    dummy12= text
    dummy12 = dummy12.replace('\r', '').replace('\n', '')
    dummyP2=dummy12.replace(' ','%20')
    dummyP3=dummyP2.replace('\'','%27')
    dummyP=dummyP3.replace('&','%26')
    part1='curl -d text='+dummyP+' http://bark.phon.ioc.ee/punctuator'
    op = commands.getstatusoutput(part1)
    output = op[1].rsplit('\n', 1)[1]
    sentsList = textParse2.sent_tokenize(output) #final sentences
    wordsList = textParse2.word_tokenize(output) #final words
    file_name = str(datetime.datetime.now().strftime("%c"))

    try:
        prescription_form2.generateFields(sentsList,wordsList,dummy12, file_name)

        MsgSignal.signal.emit(["Form Generated: /Dumps/" + file_name + ".pdf"])

        # Pop file open
        p = subprocess.Popen(["xdg-open", "./Dumps/" + file_name + ".pdf"])
        returncode = p.wait() # wait to exit

    except Exception as e:
        print("Error encountered generating form. Exception: " + str(e))
        MsgSignal.signal.emit(["Error encountered generating form. Exception: " + str(e)])

    ButtonsSignal.signal.emit([(Window.GenerateFormButton, True)])
    print("Form Generator Thread Killed.")

if __name__ == '__main__':
    pass

