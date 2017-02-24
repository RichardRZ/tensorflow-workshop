"""
reader.py

Reads, parses, and tokenizes the PTB data, and stores vectorized data in
serialized (.pik) format.
"""
import numpy as np
import pickle

TRAIN_PATH = "../data/raw_data/train.txt"
TEST_PATH = "../data/raw_data/test.txt"
OUT_PATH = "../data/processed_data/"

def parse(path, vocab=None):
    # Open File 
    with open(path, 'r') as f:
        words = f.read().split()
    
    # Create Vocabulary
    if vocab is None:
        vocab = {w: i for i, w in enumerate(list(set(words)))}

    # Return Vectorized Data
    vectorized = [vocab[w] for w in words]
    return np.array(vectorized[:-1], dtype=np.int32), np.array(vectorized[1:], dtype=np.int32), vocab


if __name__ == "__main__":
    # Parse Train File
    trX, trY, vocab = parse(TRAIN_PATH)

    # Parse Test File
    tsX, tsY, _ = parse(TEST_PATH, vocab)

    # Pickle Train Data
    with open(OUT_PATH + "train.pik", 'w') as f:
        pickle.dump((trX, trY, vocab), f)
    
    # Pickle Test Data
    with open(OUT_PATH + "test.pik", 'w') as g:
        pickle.dump((tsX, tsY, vocab), g)