import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.utils import to_categorical
import nltk
import re
import random
import os
import json
import math 


from typing import List, Tuple, Dict, Any


SEQUENCE_LENGTH: int = 100
EPOCHS: int = 10 
BATCH_SIZE: int = 128
LSTM_UNITS: int = 256
MIN_CORPUS_LENGTH: int = 500
GENERATION_LENGTH: int = 200
TEMPERATURE: float = 1.0
MODEL_FILE_PATH: str = "char_lstm_model.keras"
VOCAB_FILE_PATH: str = "char_vocab.json"
TEST_TEXT_RATIO: float = 0.1 


def clean_text_simple(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,!?;:]', '', text)
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

def sample(preds: np.ndarray, temperature: float = 1.0) -> int:
    preds = np.asarray(preds).astype('float64')
    if temperature == 0: 
        return np.argmax(preds)
    preds = np.log(preds + 1e-7) / temperature 
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def prepare_sequences(text_data: str, char_to_int_map: Dict[str, int], seq_len: int, vocab_size_for_norm: int) -> Tuple[np.ndarray | None, np.ndarray | None, int]:
    """Prepares sequences and targets from text data using a given char_to_int map."""
    sequences: List[List[int]] = []
    next_chars: List[int] = []

    filtered_text_data = ''.join([char for char in text_data if char in char_to_int_map])

    if len(filtered_text_data) <= seq_len:
        print(f"Warning: Filtered text data (length {len(filtered_text_data)}) is too short for sequence length {seq_len}.")
        return None, None, 0

    for i in range(0, len(filtered_text_data) - seq_len):
        seq_in: str = filtered_text_data[i : i + seq_len]
        seq_out: str = filtered_text_data[i + seq_len]
        sequences.append([char_to_int_map[char] for char in seq_in])
        next_chars.append(char_to_int_map[seq_out])

    n_sequences: int = len(sequences)
    if n_sequences == 0:
        return None, None, 0

    X: np.ndarray = np.reshape(sequences, (n_sequences, seq_len, 1))
    X = X / float(vocab_size_for_norm) 
    y: np.ndarray = to_categorical(next_chars, num_classes=vocab_size_for_norm)
    return X, y, n_sequences

def calculate_perplexity(model: Sequential, X_eval: np.ndarray, y_eval: np.ndarray, batch_size_eval: int) -> float:
    """Calculates perplexity given the model and evaluation data."""
    if X_eval is None or y_eval is None or X_eval.shape[0] == 0:
        print("Cannot calculate perplexity: No evaluation data provided or data is empty.")
        return float('inf')
    
    print(f"\nEvaluating model for perplexity on {X_eval.shape[0]} sequences...")
    loss, accuracy = model.evaluate(X_eval, y_eval, batch_size=batch_size_eval, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    if loss == 0: 
        perplexity = 1.0 
    elif math.isinf(loss) or math.isnan(loss):
        perplexity = float('inf')
    else:
        perplexity = np.exp(loss) 
    
    return perplexity

def main() -> None:
    model: Sequential
    char_to_int: Dict[str, int]
    int_to_char: Dict[int, str]
    vocab_size: int
    
    print("Downloading 'gutenberg' corpus from NLTK (if needed)...")
    try:
        nltk.download('gutenberg', quiet=True)
        raw_text_full: str = nltk.corpus.gutenberg.raw('carroll-alice.txt')
        print("'gutenberg' corpus ready.")
    except Exception as e:
        print(f"Could not load 'carroll-alice.txt': {e}. Exiting.")
        return

    print("Cleaning full text...")
    cleaned_full_text = clean_text_simple(raw_text_full)

    if len(cleaned_full_text) < MIN_CORPUS_LENGTH:
        print(f"Cleaned corpus is too short ({len(cleaned_full_text)} chars). Needs at least {MIN_CORPUS_LENGTH}. Exiting.")
        return

    split_idx: int = int(len(cleaned_full_text) * (1 - TEST_TEXT_RATIO))
    train_text: str = cleaned_full_text[:split_idx]
    test_text: str = cleaned_full_text[split_idx:]
    
    print(f"Total cleaned characters: {len(cleaned_full_text)}")
    print(f"Training text length: {len(train_text)} characters")
    print(f"Test text length: {len(test_text)} characters")

    if len(train_text) <= SEQUENCE_LENGTH or len(test_text) <= SEQUENCE_LENGTH:
        print("Training or test text is too short for the sequence length after splitting. Exiting.")
        return

    if os.path.exists(MODEL_FILE_PATH) and os.path.exists(VOCAB_FILE_PATH):
        print(f"Loading existing model from {MODEL_FILE_PATH}...")
        model = load_model(MODEL_FILE_PATH)
        print("Model loaded.")

        print(f"Loading vocabulary from {VOCAB_FILE_PATH}...")
        with open(VOCAB_FILE_PATH, 'r') as f:
            char_to_int = json.load(f)
        int_to_char = {i: c for c, i in char_to_int.items()}
        vocab_size = len(char_to_int)
        print(f"Vocabulary loaded. Vocab size: {vocab_size}")

        print("Preparing test data for perplexity using loaded model and vocab...")
        X_test, y_test, n_test_seq = prepare_sequences(test_text, char_to_int, SEQUENCE_LENGTH, vocab_size)
        if n_test_seq > 0:
            perplexity = calculate_perplexity(model, X_test, y_test, BATCH_SIZE)
            print(f"Perplexity on test set (loaded model): {perplexity:.2f}")
        else:
            print("Not enough test data to calculate perplexity for the loaded model.")

    else:
        print("No pre-trained model or vocabulary found. Training a new model.")
        
        print("Building vocabulary from training text...")
        chars: List[str] = sorted(list(set(train_text)))
        char_to_int = {c: i for i, c in enumerate(chars)}
        int_to_char = {i: c for i, c in enumerate(chars)}
        vocab_size = len(chars)
        print(f"Total unique characters in training data (vocab size): {vocab_size}")

        if vocab_size == 0:
            print("Vocabulary is empty (no characters in training text). Exiting.")
            return

        print(f"Saving vocabulary to {VOCAB_FILE_PATH}...")
        with open(VOCAB_FILE_PATH, 'w') as f:
            json.dump(char_to_int, f)
        print("Vocabulary saved.")

        print(f"Creating training sequences of length {SEQUENCE_LENGTH}...")
        X_train, y_train, n_train_seq = prepare_sequences(train_text, char_to_int, SEQUENCE_LENGTH, vocab_size)
        if n_train_seq == 0:
            print("No training sequences generated. Training text might be too short or filtered to empty. Exiting.")
            return
        print(f"Total training sequences: {n_train_seq}")
        print(f"Input shape (X_train): {X_train.shape}")
        print(f"Output shape (y_train): {y_train.shape}")

        print(f"Creating test sequences of length {SEQUENCE_LENGTH}...")
        X_test, y_test, n_test_seq = prepare_sequences(test_text, char_to_int, SEQUENCE_LENGTH, vocab_size)
        if n_test_seq == 0:
            print("Warning: No test sequences generated. Test text might be too short or filtered to empty. Perplexity cannot be calculated.")
        else:
            print(f"Total test sequences: {n_test_seq}")
            print(f"Input shape (X_test): {X_test.shape}")
            print(f"Output shape (y_test): {y_test.shape}")


        print("Building character-level LSTM model...")
        model = Sequential([
            LSTM(LSTM_UNITS, input_shape=(SEQUENCE_LENGTH, 1)),
            Dense(vocab_size, activation='softmax')
        ])
        model.summary()

        print("Compiling model...")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(f"Training model for {EPOCHS} epochs...")
        history: tf.keras.callbacks.History = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )

        print(f"Saving model to {MODEL_FILE_PATH}...")
        model.save(MODEL_FILE_PATH)
        print("Model saved.")

        # Calculate perplexity on the test set
        if n_test_seq > 0:
            perplexity = calculate_perplexity(model, X_test, y_test, BATCH_SIZE)
            print(f"Perplexity on test set (newly trained model): {perplexity:.2f}")
        else:
            print("Perplexity not calculated as there were no valid test sequences.")



    print("\nGenerating text...")
    
    seed_source_text = train_text if not os.path.exists(MODEL_FILE_PATH + ".trained_flag") else cleaned_full_text

    potential_seeds = []
    start_text_for_seed = cleaned_full_text #t
    
    if len(start_text_for_seed) > SEQUENCE_LENGTH:
        for i in range(0, len(start_text_for_seed) - SEQUENCE_LENGTH, SEQUENCE_LENGTH // 2): 
            seed_candidate_str = start_text_for_seed[i: i + SEQUENCE_LENGTH]
            if all(char in char_to_int for char in seed_candidate_str):
                potential_seeds.append([char_to_int[char] for char in seed_candidate_str])
            if len(potential_seeds) > 10: 
                break
    
    if not potential_seeds:
        print("Could not create any valid seed sequences from the text and loaded vocabulary.")
        default_char = list(int_to_char.values())[0] if int_to_char else 'a'
        default_char_code = char_to_int.get(default_char, 0)
        print(f"Using a fallback seed sequence of '{default_char}'s.")
        seed_sequence_int: List[int] = [default_char_code] * SEQUENCE_LENGTH
    else:
        seed_sequence_int = random.choice(potential_seeds)

    seed_sequence_char: str = ''.join([int_to_char.get(value, '?') for value in seed_sequence_int])
    print(f"--- Seed Sequence ---\n{seed_sequence_char}\n--- Generated Text ---")

    generated_text_output: str = seed_sequence_char
    current_sequence_int = list(seed_sequence_int) 

    for _ in range(GENERATION_LENGTH):
        x_pred: np.ndarray = np.reshape(current_sequence_int, (1, SEQUENCE_LENGTH, 1))
        x_pred = x_pred / float(vocab_size)

        prediction_probs: np.ndarray = model.predict(x_pred, verbose=0)[0]
        next_char_index: int = sample(prediction_probs, TEMPERATURE)
        next_char: str = int_to_char.get(next_char_index, '?')

        generated_text_output += next_char
        current_sequence_int.append(next_char_index)
        current_sequence_int = current_sequence_int[1:]

    print(generated_text_output)
    print("\n--- End of Generation ---")


    if not os.path.exists(MODEL_FILE_PATH + ".trained_flag") and os.path.exists(MODEL_FILE_PATH):
         open(MODEL_FILE_PATH + ".trained_flag", 'w').close()


if __name__ == "__main__":
    main()
  
    if os.path.exists(MODEL_FILE_PATH + ".trained_flag"):
        try:
            os.remove(MODEL_FILE_PATH + ".trained_flag")
        except OSError:
            pass