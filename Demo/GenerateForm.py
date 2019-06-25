import commands
from Form_Filling import textParse2
from Form_Filling import prescription_form2
from classes import GUISignal
import subprocess

def GenerateForm(Window, text):

    # Create Signal Objects
    FormSignal = GUISignal()
    FormSignal.signal.connect(Window.UpdateMsgBox)
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
    prescription_form2.generateFields(sentsList,wordsList,dummy12)
    FormSignal.signal.emit(["Form Generated!"])
    fileName = "023.pdf"
    p = subprocess.Popen(["xdg-open", fileName])
    returncode = p.wait() # wait to exit

    print("Form Generator Thread Killed.")


if __name__ == '__main__':
    pass

