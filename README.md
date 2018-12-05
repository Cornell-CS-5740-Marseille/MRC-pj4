# Machine Reading Comprehension

In this project, we apply Natural Language Processing knowledge to implement a Machine Reading Comprehension task. Given a paragraph and some questions, we need to judge whether the question is answerable or unanswerable. We propose three baseline systems and two final systems and compare their performance
on the dataset – Stanford Question Answering Dataset.
## Getting Started

Our project is implemented by Python 3.6

### Prerequisites

Since we use `Keras` to train the neural network models, you need to make sure the library has been installed.
If not, you can use pip to install the library.
```
pip3 install keras
```
We also use `matplotlib` to show the learning process during the training. 
```
pip3 install matplotlib
```
`pickle` is also required to save the trained model
```
pip3 install pickle
```

## Project Structure

- **data**: Training data and test data provided by TA.
- **LSTM**: Final LSTM model implemented by keras
- **MEMEN**: Final memory network implemented by keras. And an improved MEMEN based on  MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension.  
- **output**: Output kaggle format files.
- **Other Codes**:
    - baseline1.py: first baseline system
    - baseline2.py: second baseline system
    - evaluate.py: evaluate the model by precision, recall, f1 score, and accuracy
    - ngram.py: N-gram model
    - prep.py, preprocessor.py: preprocessing the text file
- README.md

## Running the Models

Explain how to run the automated tests for this system

### Baseline Systems

- run the first baseline model.
```
python baseline1.py
```
- run the second baseline model.
```
python baseline2.py
```
### Final Systems
You are able to train and evaluate the networks by commenting and uncommenting the code in `main.py`.

- train the model: uncommenting the `train()` function and run the main script.
```
python main.py
```
- evaluate the model: uncommenting the `evaluate()` function and run the main script.

- generate submitted file: uncommenting the `predict()` function and run the main script.


## Authors

* **Jialu Li** - *Baseline & Documents* - jl3855@cornell.edu
* **Charlie Wang** - *Baseline Systems* - qw248@cornell.edu
* **Ziyun Wei** - *Final Systems* - zw555@cornell.edu


## License

No license

