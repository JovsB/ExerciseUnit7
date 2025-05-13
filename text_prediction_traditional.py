import nltk
import re
import random
from collections import defaultdict, Counter

# --- Type Definitions ---
from typing import List, Dict, Tuple, DefaultDict, Set

# --- Constants ---
N_GRAM_SIZE: int = 3 # Using trigrams (predict word based on 2 previous words)
MIN_CORPUS_WORDS: int = 1000 # Minimum words to consider the corpus valid

# --- Helper Functions ---
def clean_and_tokenize(text: str) -> List[str]:
    """
    Cleans text by lowercasing, removing punctuation (except sentence-ending), 
    and tokenizing into words.
    """
    text = text.lower()
    # Keep basic sentence structure, remove other punctuation
    text = re.sub(r'[^a-z\s.?!]', '', text)

    # Tokenize by spaces and handle punctuation as separate tokens if needed
    # For simplicity, we split by space and filter empty strings.
    # NLTK word_tokenize is more robust but adds dependency for this specific part.
    tokens: List[str] = [token for token in text.split() if token]
    return tokens

# --- N-gram Model Class ---
class NGramModel:
    """
    A simple N-gram model for text prediction.
    """
    def __init__(self, n: int = 3):
        """
        Initializes the N-gram model.

        Args:
            n: The size of the N-gram (e.g., 3 for trigrams).
        """
        if n < 2:
            raise ValueError("N-gram size (n) must be at least 2.")
        self.n: int = n
        # Stores N-gram counts: (prefix_tuple) -> Counter(next_word -> count)
        self.ngram_counts: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.vocabulary: Set[str] = set()

    def train(self, tokens: List[str]) -> None:
        """
        Trains the N-gram model on a list of tokens.

        Args:
            tokens: A list of word tokens from the corpus.
        """
        if not tokens:
            print("Warning: Training with an empty token list.")
            return

        self.vocabulary.update(tokens)
        
        for i in range(len(tokens) - self.n + 1):
            ngram: Tuple[str, ...] = tuple(tokens[i : i + self.n])
            prefix: Tuple[str, ...] = ngram[:-1]
            next_word: str = ngram[-1]
            self.ngram_counts[prefix][next_word] += 1
        print(f"Trained {self.n}-gram model. Found {len(self.ngram_counts)} unique prefixes.")

    def predict_next_word(self, current_sequence: List[str]) -> str | None:
        """
        Predicts the next word based on the current sequence of words.

        Args:
            current_sequence: A list of words representing the current context.
                              The last (n-1) words will be used as the prefix.

        Returns:
            The predicted next word, or None if no prediction can be made.
        """
        if len(current_sequence) < self.n - 1:
            # Not enough context for the N-gram model
            # Fallback: could try a smaller N, or return a random frequent word
            if self.ngram_counts: # Check if model has any data
                 # Simple fallback: pick a random word from a frequent prefix if available
                all_prefixes = list(self.ngram_counts.keys())
                if all_prefixes:
                    random_prefix = random.choice(all_prefixes)
                    if self.ngram_counts[random_prefix]:
                        return self.ngram_counts[random_prefix].most_common(1)[0][0]
            return None 

        prefix: Tuple[str, ...] = tuple(current_sequence[-(self.n - 1):])

        if prefix in self.ngram_counts and self.ngram_counts[prefix]:
            # Predict the most frequent next word for this prefix
            most_common_next_words: List[Tuple[str, int]] = self.ngram_counts[prefix].most_common(1)
            return most_common_next_words[0][0]
        else:
            # Fallback if prefix not seen or has no continuations
            # Could try backoff to (N-1)-gram, or pick a random word from vocabulary
            if self.vocabulary:
                # Extremely simple fallback: return a random word from the vocabulary
                # This is not ideal for coherence but prevents errors.
                return random.choice(list(self.vocabulary))
            return None

    def generate_text(self, seed_sequence: List[str], length: int = 20) -> str:
        """
        Generates text starting with a seed sequence.

        Args:
            seed_sequence: A list of words to start the generation.
            length: The number of words to generate after the seed.

        Returns:
            A string of generated text.
        """
        if not self.ngram_counts:
            return "Model not trained. Cannot generate text."
        
        if not seed_sequence:
            # If no seed, pick a random frequent prefix to start
            print("No seed provided. Attempting to start with a random common prefix.")
            all_prefixes = [p for p, counts in self.ngram_counts.items() if counts]
            if not all_prefixes:
                return "Model has no valid starting points."
            current_sequence = list(random.choice(all_prefixes))
        else:
            current_sequence = list(seed_sequence) # Make a mutable copy

        generated_words: List[str] = list(current_sequence)

        for _ in range(length):
            # Ensure the sequence fed to predict_next_word has enough context
            context_for_prediction = current_sequence
            if len(current_sequence) < self.n -1:
                 # If current sequence is too short, we can't use the full n-gram.
                 # We could pad it, or try to pick a starting word differently.
                 # For now, we will try to predict and if it fails, break.
                 pass # Handled by predict_next_word's fallback

            next_word: str | None = self.predict_next_word(context_for_prediction)
            if next_word is None:
                # If no word can be predicted, stop generation
                # This might happen if the context is new or leads to a dead end
                print("Could not predict next word. Stopping generation.")
                break
            generated_words.append(next_word)
            current_sequence.append(next_word)
            # Keep current_sequence at the right length for the next prediction's prefix
            if len(current_sequence) >= self.n -1:
                 current_sequence = current_sequence[-(self.n -1):] 

        return ' '.join(generated_words)

# --- Main Execution ---
def main() -> None:
    """
    Main function to load data, train the N-gram model, and generate text.
    """
    try:
        print("Downloading 'gutenberg' corpus from NLTK (if not present)...")
        nltk.download('gutenberg', quiet=True)
        nltk.download('punkt', quiet=True) # For tokenization, though we use a simpler one
        print("'gutenberg' corpus ready.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("Please ensure you have an internet connection or manually download the 'gutenberg' and 'punkt' packages.")
        return

    print("Loading 'Alice in Wonderland' text...")
    try:
        alice_raw: str = nltk.corpus.gutenberg.raw('carroll-alice.txt')
    except Exception as e:
        print(f"Could not load 'carroll-alice.txt': {e}")
        print("Ensure the NLTK Gutenberg corpus is correctly installed.")
        return

    if not alice_raw or len(alice_raw.split()) < MIN_CORPUS_WORDS:
        print(f"Corpus is too short or empty (words: {len(alice_raw.split())}). Exiting.")
        return
    
    print(f"Corpus loaded. Length: {len(alice_raw)} characters.")

    print("Cleaning and tokenizing text...")
    tokens: List[str] = clean_and_tokenize(alice_raw)
    print(f"Number of tokens: {len(tokens)}")

    if len(tokens) < N_GRAM_SIZE:
        print(f"Not enough tokens ({len(tokens)}) to build a {N_GRAM_SIZE}-gram model. Exiting.")
        return

    # --- Initialize and Train Model ---
    print(f"Initializing {N_GRAM_SIZE}-gram model...")
    ngram_model = NGramModel(n=N_GRAM_SIZE)
    ngram_model.train(tokens)

    # --- Generate Text Example ---
    print("\\nGenerating text example...")
    
    # Try to find a common starting seed from the trained model
    seed_words: List[str] = []
    if ngram_model.ngram_counts:
        # Find a prefix that actually leads to predictions
        common_prefixes = [p for p, counts in ngram_model.ngram_counts.items() if counts and len(p) == ngram_model.n - 1]
        if common_prefixes:
            seed_words = list(random.choice(common_prefixes))
            print(f"Using seed: {' '.join(seed_words)}")
        else:
            print("Could not find a suitable common prefix to start generation. Using default empty seed.")
    else:
        print("Model has no data. Cannot effectively seed generation.")

    # If no good seed found, try with a very common phrase (might not be in this small corpus)
    if not seed_words and len(tokens) >= ngram_model.n -1:
        seed_words = tokens[:ngram_model.n-1] # Fallback to first few words of corpus
        print(f"Fallback seed: {' '.join(seed_words)}")
    elif not seed_words:
        print("Cannot determine a seed sequence. Text generation might be poor or fail.")


    generated_text: str = ngram_model.generate_text(seed_sequence=seed_words, length=30)
    print(f"\\nGenerated Text (starting with '{' '.join(seed_words) if seed_words else '[empty seed]'}'):")
    print(generated_text)

    # --- Interactive Prediction Example ---
    print("\\nInteractive Prediction Example:")
    # Ensure the seed has the correct length for prediction for n-gram
    if len(seed_words) >= ngram_model.n - 1:
        context_sequence: List[str] = seed_words[-(ngram_model.n - 1):]
        predicted: str | None = ngram_model.predict_next_word(context_sequence)
        print(f"Given context: '...​{' '.join(context_sequence)}'") # Zero-width space for potential line break
        print(f"Predicted next word: {predicted if predicted else '[No prediction]'}")
    else:
        print(f"Seed words list is too short for {ngram_model.n}-gram prediction.")
        # Try with a simpler context if available
        if len(tokens) >= ngram_model.n -1:
            simple_context = tokens[:ngram_model.n -1]
            predicted_simple: str | None = ngram_model.predict_next_word(simple_context)
            print(f"Trying with context: '...​{' '.join(simple_context)}'")
            print(f"Predicted next word: {predicted_simple if predicted_simple else '[No prediction]'}")


if __name__ == "__main__":
    main() 