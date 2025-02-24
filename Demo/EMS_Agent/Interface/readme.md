# Installation
install by requirements.txt. Remember to change torch and torch-geometric version according to you CUDA version. 
I'm testing my code on **CUDA 11.3** and **pytorch 1.12.1** on **Ubuntu 22.04**.

# Use
check the example in protocol_selector.py, first to initialize the model (it may take 10s, we can optimize the code
in this part to shorten initialization), then call **model(narrative)**, you will have 2 output, first is predicted protocol
and the other is probability.