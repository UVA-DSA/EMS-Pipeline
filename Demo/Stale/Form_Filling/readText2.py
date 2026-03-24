import commands
dummy12=gui2.getAudioStremText()
dummyP2=dummy12.replace(' ','%20')
dummyP=dummyP2.replace('&','%26')
part1='curl -d text='+dummyP+' http://bark.phon.ioc.ee/punctuator'
op = commands.getstatusoutput(part1)
output = op[1].rsplit('\n', 1)[1]

