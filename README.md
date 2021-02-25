# IMDB

The purpose of the project was to create a neural network to classify movie reviews from IMDB dataset. Each movie review was either positive or negative.

## Sample review from IMDB

"I'm a Petty Officer 1st Class (E-6) and have been in the USCG for 6 years and feel that this movie strongly represents the Coast Guard. There were only a few scenes that were far fetched. The most far-fetched was when PO Fischer (Kutcher) went down inside of the sinking vessel to pull the vessel's captain out of the engine room... that would never happen. Swimmers are not allowed to go inside of any vessel no matter the circumstances. Second, the Command Center (supposedly in Kodiak), it looked more like a NASA command center... we don't have any gear that hi-tech. Third, the Captain of the Airstation would not be running the search & rescue cases with like 10 people on watch. In reality it would be an E-6 or E-7 as the SAR Controller and maybe 2 other support personnel like an assist SAR Controller & a Radio Watchstander. Otherwise the movie was dead on, I think they should have incorporated more of the other rates in the CG and their roles in search & rescue instead of just Aviation based rates. Some of the scenes from "A" school reminded me of my days their and the dumb stuff I did and got in trouble for in my younger days."

## dataprocessing.py

File contains scripts used to process the data from a set of txt-files (each containing one review) into a dataframe object that can be used for training the model. 

## lstm-cell.py

This file contains the model built with LSTMCell objects in order to better control the data flow, as cells lack internal iteration loops. This can be especially useful when dealing with cases were we don't necessarily want to iterate over whole time dimension in one go. All characters were converted into lowercase and interpunction has been stripped. Next the whole review dataset is concatenated into one string and converted into a bag of words. Four special tokens are added (start of string, end of string, unknown, padding). Then the bag of words is converted into a set to remove duplicates. Next all elements of the set are appended into a dictionary and given unique IDs. All input reviews are connected with SOS token at the beggining, EOS at the end, and padded/cut to length of 20. Resulting objects are encoded into integers with the use of dictionary prepared earlier.

The model's first layer is embedding (it converts integer IDs into word vectors). Next there are two LSTM layers and after that there are two linear layers, activated by gelu function. The last layer is linear layer activated by sigmoid function.

## lstm.py

This file contains the model built with LSTM objects which were used to automate a part of code compared to LSTMCells. The data is prepared the same as in lstm-cell.py file. 

The models's first layer is embedding (it converts integer IDs into word vectors). Next there is LSTM layer, and after that there is dropout and normalize layer. Next there are two linear layers activated by gelu function after each there is a dropout layer. The last layer is linear activated by sigmoid. 


## Technologies
* Python - version 3.8.3
* Numpy - version 1.18.5
* Pandas - version 1.0.5
* PyTorch - version 1.7.0 CUDA 10.1
* PyTorch-Lightning - version 1.0.6
