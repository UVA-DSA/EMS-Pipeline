from nltk.tokenize import sent_tokenize, word_tokenize
import io
##file1 = open("/Users/sarahmasudpreum/Downloads/nonImperativeAdvice.txt", "r")
#### print file1.read()

##L = list()
##for line in file1:
##    print(line)
##    #for sentence in sent_tokenize(line):
##    if line not in L:
##        L.append(line)
##        #print(sentence)
##for item in L:
##    wfile.write(item)
##file1.close()
##wfile.close()
#wfile = open("nonImperativeAdvice_output_sentenceLevel.txt","w")
with io.open("/Users/sarahmasudpreum/Dropbox/nonImperativeAdvice_output_sentenceLevel.txt", "w", encoding="utf-8") as wfile:
    with io.open("/Users/sarahmasudpreum/Dropbox/nonImperativeAdvice_output.txt", "r", encoding="utf-8") as file1:
    #file1 = open("/Users/sarahmasudpreum/Dropbox/nonImperativeAdvice_output.txt", "r")
        for line in file1:
            #print(line)
            for sentence in sent_tokenize(line):
                print(sentence)
                wfile.write(sentence+"\n")                
            
file1.close()
wfile.close()

##
##EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
##
###print(sent_tokenize(EXAMPLE_TEXT))
##for sentence in sent_tokenize(EXAMPLE_TEXT):
##    print(sentence)
##    #print("\t")
##	  
##print(word_tokenize(EXAMPLE_TEXT))
##for token in word_tokenize(EXAMPLE_TEXT):
##    print(token)

