import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import nltk
import re
import random
import os # Added for file operations
import json # Added for saving/loading vocabulary

# --- Type Definitions ---
from typing import List, Tuple, Dict, Any

# --- Constants ---
SEQUENCE_LENGTH: int = 100  # Length of input character sequences
EPOCHS: int = 10  # Number of epochs for training (can be increased for better results)
BATCH_SIZE: int = 128
LSTM_UNITS: int = 256 # Increased units for character-level model
MIN_CORPUS_LENGTH: int = 500 # Minimum characters needed
GENERATION_LENGTH: int = 200 # How many characters to generate
TEMPERATURE: float = 1.0 # Controls randomness in generation (higher -> more random)
MODEL_FILE_PATH: str = "char_lstm_model.keras" # Path to save/load the model
VOCAB_FILE_PATH: str = "char_vocab.json" # Path to save/load vocabulary

# --- Helper Functions ---
def clean_text_simple(text: str) -> str:
    """
    Simple cleaning: lowercase and keep basic characters.
    """
    text = text.lower()
    # Keep letters, numbers, basic punctuation, and spaces
    text = re.sub(r'[^a-z0-9\\s.,!?;:]', '', text)
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# Sample text from the distribution
def sample(preds: np.ndarray, temperature: float = 1.0) -> int:
    """Helper function to sample an index from a probability array."""
    # Cast preds to float64 to avoid precision issues
    preds = np.asarray(preds).astype('float64')
    # Apply temperature scaling
    preds = np.log(preds + 1e-7) / temperature # Add epsilon to avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # Sample using multinomial distribution
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# --- Main Execution ---
def main() -> None:
    """
    Main function for character-level LSTM text prediction.
    """
    model: Sequential
    char_to_int: Dict[str, int]
    int_to_char: Dict[int, str]
    vocab_size: int
    text: str # Define text here for broader scope if loading vocab

    # --- Load and Prepare Data ---
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

        # We still need the text for generating seed sequences, though not for training
        print("Downloading 'gutenberg' corpus from NLTK (if needed for seed)...")
        try:
            nltk.download('gutenberg', quiet=True)
            print("'gutenberg' corpus ready.")
            raw_text: str = nltk.corpus.gutenberg.raw('carroll-alice.txt')
            text = clean_text_simple(raw_text) # Cleaned text for consistent seed generation
            if len(text) < SEQUENCE_LENGTH: # Ensure text is long enough for at least one sequence
                print(f"Corpus text is too short ({len(text)} chars) to generate a seed. Exiting.")
                return
        except Exception as e:
            print(f"Could not load 'carroll-alice.txt' for seed generation: {e}. Exiting.")
            return

    else:
        print("No pre-trained model or vocabulary found. Training a new model.")
        try:
            print("Downloading 'gutenberg' corpus from NLTK (if needed)...")
            nltk.download('gutenberg', quiet=True)
            print("'gutenberg' corpus ready.")
        except Exception as e:
            print(f"Error downloading NLTK data: {e}. Exiting.")
            return

        print("Loading 'Alice in Wonderland' text...")
        try:
            raw_text: str = nltk.corpus.gutenberg.raw('carroll-alice.txt')
        except Exception as e:
            print(f"Could not load 'carroll-alice.txt': {e}. Exiting.")
            return

        print("Cleaning text...")
        text = clean_text_simple(raw_text)

        if len(text) < MIN_CORPUS_LENGTH:
            print(f"Cleaned corpus is too short ({len(text)} chars). Needs at least {MIN_CORPUS_LENGTH}. Exiting.")
            return

        print(f"Corpus length: {len(text)} characters.")

        # Create character mapping
        chars: List[str] = sorted(list(set(text)))
        char_to_int = {c: i for i, c in enumerate(chars)}
        int_to_char = {i: c for i, c in enumerate(chars)}
        vocab_size = len(chars)
        print(f"Total unique characters (vocab size): {vocab_size}")

        # Save vocabulary
        print(f"Saving vocabulary to {VOCAB_FILE_PATH}...")
        with open(VOCAB_FILE_PATH, 'w') as f:
            json.dump(char_to_int, f)
        print("Vocabulary saved.")

        # Create sequences and targets
        print(f"Creating sequences of length {SEQUENCE_LENGTH}...")
        sequences: List[List[int]] = []
        next_chars: List[int] = []
        for i in range(0, len(text) - SEQUENCE_LENGTH):
            seq_in: str = text[i : i + SEQUENCE_LENGTH]
            seq_out: str = text[i + SEQUENCE_LENGTH]
            sequences.append([char_to_int[char] for char in seq_in])
            next_chars.append(char_to_int[seq_out])

        n_sequences: int = len(sequences)
        print(f"Total sequences: {n_sequences}")

        if n_sequences == 0:
            print("No sequences generated. Text might be shorter than SEQUENCE_LENGTH. Exiting.")
            return

        # Reshape sequences for LSTM input: [samples, time_steps, features]
        X: np.ndarray = np.reshape(sequences, (n_sequences, SEQUENCE_LENGTH, 1))
        X = X / float(vocab_size) # Normalize
        y: np.ndarray = to_categorical(next_chars, num_classes=vocab_size)

        print(f"Input shape (X): {X.shape}")
        print(f"Output shape (y): {y.shape}")

        # --- Build the LSTM Model ---
        print("Building character-level LSTM model...")
        model = Sequential([
            LSTM(LSTM_UNITS, input_shape=(SEQUENCE_LENGTH, 1)),
            Dense(vocab_size, activation='softmax')
        ])

        model.summary()

        # --- Compile the Model ---
        print("Compiling model...")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # --- Train the Model ---
        print(f"Training model for {EPOCHS} epochs...")
        history: tf.keras.callbacks.History = model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )

        # --- Save the Model ---
        print(f"Saving model to {MODEL_FILE_PATH}...")
        model.save(MODEL_FILE_PATH)
        print("Model saved.")

    # --- Generate Text (Common for both loaded and newly trained model) ---
    print("\nGenerating text...")

    # Create seed sequences from the current 'text' (either freshly loaded or from training part)
    # This ensures consistency in how seeds are chosen.
    sequences_for_seed: List[List[int]] = []
    if len(text) > SEQUENCE_LENGTH: # Check if text is long enough
        for i in range(0, len(text) - SEQUENCE_LENGTH):
            seq_in: str = text[i : i + SEQUENCE_LENGTH]
            # Ensure all characters in seq_in are in char_to_int before converting
            if all(char in char_to_int for char in seq_in):
                sequences_for_seed.append([char_to_int[char] for char in seq_in])
            # else:
            #     print(f"Skipping seed candidate due to unknown char: {seq_in}")


    if not sequences_for_seed:
        print("Could not create any valid seed sequences from the text and loaded vocabulary.")
        print("This might happen if the text used for seed generation contains characters not in the loaded vocabulary.")
        # Fallback: create a dummy seed if no valid seed sequence can be formed
        print("Using a fallback seed sequence of 'a's.")
        first_char_code = char_to_int.get('a', 0) # Get 'a' or default to 0
        seed_sequence_int: List[int] = [first_char_code] * SEQUENCE_LENGTH
        if 'a' not in char_to_int and 0 not in int_to_char: # if 'a' and default 0 are not in vocab
             # try to get the first available character from the vocabulary
            if int_to_char:
                first_available_char_code = list(int_to_char.keys())[0]
                seed_sequence_int = [first_available_char_code] * SEQUENCE_LENGTH
            else:
                print("Vocabulary is empty. Cannot generate text. Exiting.")
                return

    else:
        start_index: int = random.randint(0, len(sequences_for_seed) - 1)
        seed_sequence_int = sequences_for_seed[start_index]

    seed_sequence_char: str = ''.join([int_to_char.get(value, '?') for value in seed_sequence_int]) # Use .get for safety

    print(f"--- Seed Sequence ---\n{seed_sequence_char}\n--- Generated Text ---")

    generated_text: str = seed_sequence_char
    current_sequence_int = list(seed_sequence_int)

    for _ in range(GENERATION_LENGTH):
        x_pred: np.ndarray = np.reshape(current_sequence_int, (1, SEQUENCE_LENGTH, 1))
        x_pred = x_pred / float(vocab_size) # Normalize

        prediction_probs: np.ndarray = model.predict(x_pred, verbose=0)[0]
        next_char_index: int = sample(prediction_probs, TEMPERATURE)

        next_char: str = int_to_char.get(next_char_index, '?') # Use .get for safety

        generated_text += next_char
        current_sequence_int.append(next_char_index)
        current_sequence_int = current_sequence_int[1:]

    print(generated_text)
    print("\n--- End of Generation ---")

if __name__ == "__main__":
    main()