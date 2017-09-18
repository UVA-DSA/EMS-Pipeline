''' 
Basic cleaning and pre-processing of EMS narratives

Copyright (C) 2017 University of Virginia, Homa Alemzadeh

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
import nltk
from nltk.collocations import *
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import bigrams
from nltk import trigrams
from nltk.util import ngrams
#from nltk.tag.stanford import StanfordPOSTagger
import csv, re, math, operator, sys, os, glob
from time import time

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
#pos = StanfordPOSTagger('./stanford-postagger-2013-06-20/models/english-left3words-distsim.tagger', 
#                './stanford-postagger-2013-06-20/stanford-postagger.jar')

# N-gram patterns for technical terms
patterns = [['NN'],['JJ'], ['JJ','JJ'],['JJ', 'NN'], ['NN','NN'], ['JJ', 'JJ', 'NN'], ['JJ', 'NN', 'NN'], 
            ['NN', 'JJ', 'NN'], ['NN', 'NN', 'NN'],['NN', 'IN', 'NN'],['JJ','JJ','JJ'],
			['JJ', 'JJ', 'JJ','NN'], ['JJ', 'JJ', 'NN','NN'], ['JJ', 'NN', 'NN','NN'],['JJ', 'NN', 'JJ','NN'],['NN', 'JJ', 'JJ','NN'], 
			['NN', 'NN', 'NN','NN'],['NN','JJ', 'NN','NN'], ['NN', 'NN','JJ', 'NN']]
# For filtering junk
non_tech_words = []

# Tokenization
def get_tokens(text):
	stops = set(stopwords.words('english'))
	regex = re.compile('[%s]' % re.escape('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))

	# Get the sentences
	sentences = regex.sub(' ', text)
	
	# Get the words
	raw_tokens = list(set(word_tokenize(unicode(sentences, errors='ignore'))))
	
	# Filter numbers and characters
	#tokens= [str(t) for t in raw_tokens if str(t) not in stops and not str(t).isdigit() and len(str(t))>1]
	tokens= [str(t) for t in raw_tokens if not str(t).isdigit() and len(str(t))>1]
	tokens = [t for t in tokens if t not in non_tech_words and t.isalpha()]
	
	return tokens
	
# Part of Speech Tagging 
def get_pos_tags(tokens):
	regex = re.compile('[%s]' % re.escape('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
	tags = [];
	starti = 0
	endi = 0
	no_chunks = len(tokens)/5000+1;
	print 'Process '+str(len(tokens))+' tokens in '+str(no_chunks)+ ' chunks..' 
	for l in range(0, no_chunks):
		endi =  min((starti + (len(tokens)/no_chunks) ), len(tokens))
		print "Tagging #" + str(l) + ": from " + str(starti)+ " to "+str(endi-1)
		tags = tags + nltk.pos_tag(tokens[starti:endi]);
		#tags = tags + pos.tag(tokens[starti:endi])[0]; 		
		starti = endi;
	
	print str(len(tags))+" words tagged.."

	# Save all the Noun and Adjective unigrams in a hash table
	tag_set = {'Word':'Tag'}
	for tag in tags:
		if (cleanseNN([str(tag[1])]) in patterns[0:2]):
			tag_set[str(tag[0])] = str(tag[1])		
		#print cleanseNN([str(tag[1])])
	#print tags
	#print '\n'
	return tag_set

# Clean part of speech tags
def cleanseNN(list):
	for i in range(0, len(list)):
		for k in range(0, len(list[i])):
			if("NN" in list[i][k]):
				list[i][k] = "NN"
	return list
	
# Look for longest n-gram appearing in each sentence with the patterns of technical terms
def get_tech_ngrams(text, tag_set):	
	# Normalization and Punctuation filtering=> keep sentence separators
	text = text.lower()
	sentences = re.split('\.(?!\d)', text)
	#print sentences
	
	results = {'ngram':'tags'};
	n_gram = []
	tags = []
	n_gram_str = ''
	tag_str = ''

	for s in sentences:
		regex = re.compile('[%s]' % re.escape('!"#$%&\'()*+/:<=>?@[\\]^_`{|}~-'))
		Text = regex.sub(' ',s)
		# Get the words
		words = word_tokenize(unicode(s, errors='ignore'))
		w_i = -1;
		# Filter numbers 
		#words= [str(t) for t in raw_tokens if not str(t).isdigit() and len(str(t))>1]
	
		#print '--->' + Text + '\n'
		s_result = []
		for w in words:
			w_i = w_i + 1;
			if (tag_set.has_key(w)):
				n_gram.append(w)
				tags.append(tag_set[w])
				#print "n-gram = "+n_gram[-1]
			#print w+'-'+str(words.index(w))
			# If this is the last word in the list or a non-NJ word, we finalize the n-gram
			if not(tag_set.has_key(w)) or (w_i == len(words)-1):
				# Only if we found something
				if (len(n_gram) > 1):
					#print "long n-gram = "+n_gram[0]
					# If the pattern of tags is of interest, save it
					if (cleanseNN(tags) in patterns):
						n_gram_str = ' '.join(n_gram)
						tag_str = ', '.join(tags)
						if not(n_gram_str in s_result) :
							s_result.append(n_gram_str)
				n_gram_str = n_gram_str.decode('utf-8')
				if (n_gram_str in Text):
					if not(results.has_key(n_gram_str)):
						results[n_gram_str] = tag_str
				else:
					print 'ngram not found in text: '+n_gram_str
				# Restart searching for next n-gram
				n_gram = []
				tags = []
				n_gram_str = ''
				tag_str = ''

	print str(len(results.keys()))+" n-grams found.."	
	return results

def main():
	os.chdir("./dataset")
	
	# Set default encoding of python to utf8
	reload(sys)  
	sys.setdefaultencoding('utf8')
	
	with open('output.csv', 'w') as output:
		csvwriter = csv.writer(output)
		csvwriter.writerow(["Data File", "Narrative", "N-grams"])
		for file in glob.glob("*.txt"):
			with open(file, 'r') as reader:
				text = ''
				for line in reader:
					text = text + line.rstrip('\n\r').lower()
				
				print "\nProcessing "+file
				# Tokenization
				tokens = get_tokens(text)

				# Part of speech Tagging
				tag_set = get_pos_tags(tokens)
				
				# Technical N-gram extraction
				ngrams = get_tech_ngrams(text, tag_set)
				
				# Write to output
				csvwriter.writerow([file, text, ngrams.keys()])

if __name__ == '__main__':
    sys.exit(main())

