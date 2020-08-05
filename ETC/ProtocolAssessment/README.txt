A- TROUBLESHOOTING TIPS: (add to this as you find and fix problems)
1) Check file paths. You may need to update based on how you're running the code.

2) jupyter notebook, Python 3 kernel
sklearn==0.0
scipy==1.3.0
pandas==0.24.2
numpy==1.16.2
nltk==3.4.3
matplotlib==3.0.3
openpyxl==3.0.3

if openpyxl isn't found in weight_concepts.ipynb, uncomment the following code in the imports the first time you run that cell in a session:
import sys
!{sys.executable} -m pip install openpyxl

3) .xlsx files are locked and unreadable by Jupyter Notebook if changed on the computer. Just rename any .xlsx file that you need to .xls and it should work. Or make it a csv. Just don't forget to change the extension name in the filepath in the code if you change it locally. 

4) if you run into trouble while executing, try uncommenting some of the print statements and see if you get what you're expecting

5) the concept extractor code in ProtocolAssessment.ipynb was originally written by Sile Shu and is in behaviours_m.py and ConceptExtractor.py in the UVA-DSA EMS-Pipeline github repo. All other code was originally written by Elizabeth Shelton. Message Elizabeth on Slack if you have questions. 



B- ALL FILES AS OF 06/11/20 AND THEIR IMPORTANCE:
CLfromVt.csv: concepts needed for cosine similarity in ProtocolAssessment.ipynb

concept_list(s&s)_revised.csv: concepts needed for cosine similarity in ProtocolAssessment.ipynb; to add concepts, add to left column at the bottom. No CUI needed.

equiv_classes.txt: created by Protocol Clustering.ipynb. Current data overwritten when that notebook is run again. A text representation of the dictionary created, where each entry contains the protocols for that cluster.

ODEMSA_Protocols_Expanded.xlsx: 36 protocols with their related concepts. Not weighted. Used by weight_concepts.ipynb to create ODEMSA_Protocols_Expanded_weighted.xls. To add new protocols, new concepts, or update weights, modify this file and run weight_concepts.ipynb again. May need to be converted to a .xls file (just rename the file extension).

ODEMSA_Protocols_Expanded_weighted.xls: Created by weight_concepts.ipynb. Current data overwritten when that notebook is run again. All protocols and concepts, but each concept is weighted, where a higher-weighted concept is rarer (and therefore more useful in designating a concept). Used in ProtocolAssessment.ipynb.

ODEMSA_Protocols_Expanded_weighted.xlsx: The original file for the .xls. Not used in any program here. .xlsx files are locked and unreadable by Jupyter Notebook if changed on the computer. Just rename any .xlsx file that you need to .xls and it should work. 

Protocol Clustering.ipynb: uses scikit-learn's Affinity Propagation (at the time of writing this, 06/11/20) to cluster the cosine matrix in protocol_results_weighted_all.csv created by ProtocolAssessment.ipynb. Updates equiv_classes.txt when run.

protocol_results.csv: The original protocol results without weighted concepts.

protocol_results_weighted.csv: The protocol results with weighted concepts, but without all concepts updated

protocol_results_weighted_all.csv: The most up-to-date version of the cosine matrix of protocols' cosign similarities with each other. Created by ProtocolAssessment.ipynb; used in Protocol Clustering.ipynb to produce equiv_classes.txt.

ProtocolAssessment.ipynb: Uses CLfromVT.csv, concept_list(s&s)_revised.csv, and ODEMSA_Protocols_Expanded_weighted.xls to calculate the cosign similarities between each protocol. Outputs this into a matrix in protocol_results_weighted_all.csv where row 0 and column 0 are the protocol names. 

weight_concepts.ipynb: Uses ODEMSA_Protocols_Expanded.xlsx to weight the concepts, outputting ODEMSA_Protocols_Expanded_weighted.xls. To change how concepts are weighted, modify this file. Currently the weights are hardcoded in based on a graph that is produced and range from 1-5; may be able to use metrics from that analysis to weight the concepts in a more automated way. 