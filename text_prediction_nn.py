import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import nltk
import re
import random

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
    # --- Load and Prepare Data ---
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
    text: str = clean_text_simple(raw_text)

    if len(text) < MIN_CORPUS_LENGTH:
        print(f"Cleaned corpus is too short ({len(text)} chars). Needs at least {MIN_CORPUS_LENGTH}. Exiting.")
        return

    print(f"Corpus length: {len(text)} characters.")

    # Create character mapping
    chars: List[str] = sorted(list(set(text)))
    char_to_int: Dict[str, int] = {c: i for i, c in enumerate(chars)}
    int_to_char: Dict[int, str] = {i: c for i, c in enumerate(chars)}
    vocab_size: int = len(chars)
    print(f"Total unique characters (vocab size): {vocab_size}")

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
    # Here, features = 1 because we have one character index per time step.
    X: np.ndarray = np.reshape(sequences, (n_sequences, SEQUENCE_LENGTH, 1))
    # Normalize input integer values to be between 0 and 1 (helps LSTM)
    X = X / float(vocab_size)
    # One-hot encode the output variable (the next character)
    y: np.ndarray = to_categorical(next_chars, num_classes=vocab_size)

    print(f"Input shape (X): {X.shape}") # (num_sequences, sequence_length, 1)
    print(f"Output shape (y): {y.shape}") # (num_sequences, vocab_size)

    # --- Build the LSTM Model ---
    print("Building character-level LSTM model...")
    model: Sequential = Sequential([
        # Input shape is (sequence_length, 1) - features dimension
        LSTM(LSTM_UNITS, input_shape=(SEQUENCE_LENGTH, 1)),
        # Output layer: Dense with softmax activation for probability distribution over chars
        Dense(vocab_size, activation='softmax')
    ])

    model.summary()

    # --- Compile the Model ---
    print("Compiling model...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # --- Train the Model ---
    print(f"Training model for {EPOCHS} epochs...")
    # Consider adding callbacks like ModelCheckpoint to save the best model
    history: tf.keras.callbacks.History = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # --- Generate Text ---
    print("\\nGenerating text...")
    # Select a random seed sequence from the original sequences
    start_index: int = random.randint(0, n_sequences - 1)
    seed_sequence_int: List[int] = sequences[start_index]
    seed_sequence_char: str = ''.join([int_to_char[value] for value in seed_sequence_int])
    
    print(f"--- Seed Sequence ---\\n{seed_sequence_char}\\n--- Generated Text ---")

    generated_text: str = seed_sequence_char # Start with the seed

    # Generate characters one by one
    current_sequence_int = list(seed_sequence_int) # Use list for manipulation
    for _ in range(GENERATION_LENGTH):
        # Prepare the input for the model
        x_pred: np.ndarray = np.reshape(current_sequence_int, (1, SEQUENCE_LENGTH, 1))
        x_pred = x_pred / float(vocab_size) # Normalize

        # Get prediction probabilities
        prediction_probs: np.ndarray = model.predict(x_pred, verbose=0)[0]
        
        # Sample the next character index based on probabilities and temperature
        next_char_index: int = sample(prediction_probs, TEMPERATURE)
        
        # Convert index back to character
        next_char: str = int_to_char[next_char_index]
        
        # Append predicted character to generated text
        generated_text += next_char
        
        # Update the sequence for the next prediction: remove first char, append new char
        current_sequence_int.append(next_char_index)
        current_sequence_int = current_sequence_int[1:] # Slide window

    print(generated_text)
    print("\\n--- End of Generation ---")

if __name__ == "__main__":
    main() 