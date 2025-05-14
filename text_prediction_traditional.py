import nltk
import re
import random
from collections import defaultdict, Counter
import math 

from typing import List, Dict, Tuple, DefaultDict, Set


N_GRAM_SIZE: int = 3 
MIN_CORPUS_WORDS: int = 1000 
TEST_SET_RATIO: float = 0.2 

def clean_and_tokenize(text: str) -> List[str]:
    """
    Cleans text by lowercasing, removing punctuation (except sentence-ending),
    and tokenizing into words.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s.?!]', '', text)

    tokens: List[str] = [token for token in text.split() if token]
    return tokens

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
        self.ngram_counts: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.prefix_totals: Counter[Tuple[str, ...]] = Counter()
        self.vocabulary: Set[str] = set()

    def train(self, tokens: List[str]) -> None:
        """
        Trains the N-gram model on a list of tokens.

        Args:
            tokens: A list of word tokens from the training corpus.
        """
        if not tokens:
            print("Warning: Training with an empty token list.")
            return

        self.vocabulary.update(tokens) 

        if len(tokens) < self.n:
            print(f"Warning: Number of tokens ({len(tokens)}) is less than N-gram size ({self.n}). Model may not train effectively.")
           
        for i in range(len(tokens) - self.n + 1):
            ngram: Tuple[str, ...] = tuple(tokens[i : i + self.n])
            prefix: Tuple[str, ...] = ngram[:-1]
            next_word: str = ngram[-1]
            self.ngram_counts[prefix][next_word] += 1
            self.prefix_totals[prefix] += 1
        print(f"Trained {self.n}-gram model. Vocabulary size: {len(self.vocabulary)}. Found {len(self.ngram_counts)} unique prefixes.")

    def get_word_probability(self, prefix: Tuple[str, ...], next_word: str) -> float:
        """
        Calculates the probability of next_word given prefix using Add-1 Smoothing.
        P(next_word | prefix) = (count(prefix, next_word) + 1) / (count(prefix) + |V|)

        Args:
            prefix: The context (n-1 words).
            next_word: The word whose probability we want to calculate.

        Returns:
            The smoothed probability of the next_word.
        """
        if not self.vocabulary: 
            return 0.0

        vocab_size: int = len(self.vocabulary)
        next_word_count: int = self.ngram_counts[prefix][next_word]
        prefix_count: int = self.prefix_totals[prefix]

        return (next_word_count + 1) / (prefix_count + vocab_size)

    def calculate_perplexity(self, test_tokens: List[str]) -> float:
        """
        Calculates the perplexity of the model on a given test set.
        Perplexity = 2^(-1/N * sum(log2(P(word_i | context_i))))
        Uses Add-1 Smoothing for probability calculation.

        Args:
            test_tokens: A list of word tokens from the test set.

        Returns:
            The perplexity score. Returns float('inf') if calculation is impossible.
        """
        if not self.vocabulary:
            print("Warning: Vocabulary is empty. Cannot calculate perplexity.")
            return float('inf')
        if len(test_tokens) < self.n:
            print(f"Warning: Test set ({len(test_tokens)} tokens) is shorter than N-gram size ({self.n}). Perplexity might not be meaningful.")
            

        log_likelihood_sum: float = 0.0
        num_evaluated_words: int = 0

        for i in range(self.n - 1, len(test_tokens)):
            prefix: Tuple[str, ...] = tuple(test_tokens[i - (self.n - 1) : i])
            actual_next_word: str = test_tokens[i]

            prob: float = self.get_word_probability(prefix, actual_next_word)

            if prob > 0:
                log_likelihood_sum += math.log2(prob)
                num_evaluated_words += 1
            else:
                print(f"Warning: Probability of '{actual_next_word}' given '{' '.join(prefix)}' is {prob}. Skipping this word in perplexity.")


        if num_evaluated_words == 0:
            print("Warning: No words could be evaluated for perplexity (e.g., test set too short or all probabilities were zero).")
            return float('inf')

        cross_entropy: float = -log_likelihood_sum / num_evaluated_words
        perplexity: float = math.pow(2, cross_entropy)
        return perplexity


    def predict_next_word(self, current_sequence: List[str]) -> str | None:
        """
        Predicts the next word based on the current sequence of words.
        Uses simple most_common, does not use smoothing for prediction itself here.
        """
        if len(current_sequence) < self.n - 1:
            if self.ngram_counts:
                all_prefixes = list(self.ngram_counts.keys())
                if all_prefixes:
                    random_prefix = random.choice(all_prefixes)
                    if self.ngram_counts[random_prefix]:
                        return self.ngram_counts[random_prefix].most_common(1)[0][0]
            return None

        prefix: Tuple[str, ...] = tuple(current_sequence[-(self.n - 1):])

        if prefix in self.ngram_counts and self.ngram_counts[prefix]:
            most_common_next_words: List[Tuple[str, int]] = self.ngram_counts[prefix].most_common(1)
            return most_common_next_words[0][0]
        else:
            if self.vocabulary:
                return random.choice(list(self.vocabulary))
            return None

    def generate_text(self, seed_sequence: List[str], length: int = 20) -> str:
        """
        Generates text starting with a seed sequence.
        """
        if not self.ngram_counts and not self.vocabulary: 
            return "Model not trained or vocabulary is empty. Cannot generate text."

        current_sequence = list(seed_sequence) 

        if not current_sequence:
            print("No seed provided. Attempting to start with a random common prefix.")
            valid_prefixes = [p for p in self.prefix_totals.keys() if self.prefix_totals[p] > 0 and len(p) == self.n -1]
            if valid_prefixes:
                current_sequence = list(random.choice(valid_prefixes))
            elif self.vocabulary:
                 current_sequence = random.sample(list(self.vocabulary), min(len(self.vocabulary), self.n -1))
            else: 
                return "Model has no data or valid starting points."


        generated_words: List[str] = list(current_sequence)

        for _ in range(length):
            context_for_prediction = current_sequence

            next_word: str | None = self.predict_next_word(context_for_prediction)
            if next_word is None:
            
                if self.vocabulary:
                    print("Could not predict next word with primary strategy, picking random from vocab.")
                    next_word = random.choice(list(self.vocabulary))
                else:
                    print("Could not predict next word and vocabulary is empty. Stopping generation.")
                    break
            if next_word is None:
                 print("Stopping generation as no word could be chosen.")
                 break

            generated_words.append(next_word)
            current_sequence.append(next_word)
           
            if len(current_sequence) >= self.n -1: 
                current_sequence = current_sequence[-(self.n -1):]


        return ' '.join(generated_words)

def main() -> None:
    """
    Main function to load data, train the N-gram model, generate text, and calculate perplexity.
    """
    try:
        print("Downloading 'gutenberg' corpus from NLTK (if not present)...")
        nltk.download('gutenberg', quiet=True)
        nltk.download('punkt', quiet=True)
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
    print(f"Total number of tokens: {len(tokens)}")

    if len(tokens) < N_GRAM_SIZE * 2:
        print(f"Not enough tokens ({len(tokens)}) for a meaningful train/test split and to build a {N_GRAM_SIZE}-gram model. Exiting.")
        return

    split_index: int = int(len(tokens) * (1 - TEST_SET_RATIO))
    train_tokens: List[str] = tokens[:split_index]
    test_tokens: List[str] = tokens[split_index:]

    print(f"Training set size: {len(train_tokens)} tokens")
    print(f"Test set size: {len(test_tokens)} tokens")

    if len(train_tokens) < N_GRAM_SIZE:
         print(f"Training set too small ({len(train_tokens)} tokens) to build a {N_GRAM_SIZE}-gram model. Exiting.")
         return
    if len(test_tokens) < N_GRAM_SIZE:
        print(f"Warning: Test set is very small ({len(test_tokens)} tokens). Perplexity might not be reliable.")


    print(f"\nInitializing {N_GRAM_SIZE}-gram model...")
    ngram_model = NGramModel(n=N_GRAM_SIZE)
    print("Training model...")
    ngram_model.train(train_tokens)

    if ngram_model.vocabulary and len(test_tokens) >= ngram_model.n :
        print("\nCalculating perplexity on the test set...")
        perplexity = ngram_model.calculate_perplexity(test_tokens)
        print(f"Perplexity on test set: {perplexity:.2f}")
    else:
        print("\nSkipping perplexity calculation due to insufficient test data or untrained model.")

    print("\nGenerating text example...")

    seed_words: List[str] = []
    
    if ngram_model.prefix_totals:
    
        common_prefixes = [p for p, count in ngram_model.prefix_totals.items() if count > 0 and len(p) == ngram_model.n - 1]
        if common_prefixes:
            seed_words = list(random.choice(common_prefixes))
            print(f"Using seed from common prefixes: '{' '.join(seed_words)}'")
        else:
             print("Could not find a suitable common prefix of required length.")
    
    if not seed_words: 
        if len(train_tokens) >= ngram_model.n -1:
            seed_words = train_tokens[:ngram_model.n-1] 
            print(f"Using fallback seed from start of training data: '{' '.join(seed_words)}'")
        elif ngram_model.vocabulary: 
            seed_words = random.sample(list(ngram_model.vocabulary), min(len(ngram_model.vocabulary), max(1, ngram_model.n-1)))
            print(f"Using fallback seed from vocabulary: '{' '.join(seed_words)}'")
        else:
            print("Cannot determine a seed sequence. Text generation might be poor or fail.")


    generated_text: str = ngram_model.generate_text(seed_sequence=seed_words, length=30)
    print(f"\nGenerated Text (starting with '{' '.join(seed_words) if seed_words else '[empty/default seed]'}'):")
    print(generated_text)

    print("\nInteractive Prediction Example:")
    if seed_words and len(seed_words) >= ngram_model.n - 1:
        context_sequence: List[str] = seed_words[-(ngram_model.n - 1):]
        predicted: str | None = ngram_model.predict_next_word(context_sequence)
        print(f"Given context: '...{' '.join(context_sequence)}'")
        print(f"Predicted next word: {predicted if predicted else '[No prediction]'}")
    elif len(train_tokens) >= ngram_model.n -1 :
        simple_context = train_tokens[:ngram_model.n -1]
        predicted_simple: str | None = ngram_model.predict_next_word(simple_context)
        print(f"No suitable seed for interactive example, trying with context: '...{' '.join(simple_context)}'")
        print(f"Predicted next word: {predicted_simple if predicted_simple else '[No prediction]'}")
    else:
        print(f"Not enough data for a {ngram_model.n}-gram prediction context.")


if __name__ == "__main__":
    main()