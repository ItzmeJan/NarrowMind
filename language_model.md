# LanguageModel API Documentation

Complete reference for the `LanguageModel` struct and its public methods.

## Overview

`LanguageModel` is a statistical language model that uses n-grams, TF-IDF vectors, and modern sampling techniques for question-answering and text generation.

## Struct

```rust
pub struct LanguageModel {
    pub ngram_sizes: Vec<usize>,           // N-gram sizes being used (e.g., [2, 3])
    pub ngram_weights: HashMap<usize, f64>, // Weights for each n-gram size
    pub vocabulary: Vec<String>,          // All words in the vocabulary
    // ... (internal fields)
}
```

### Public Fields

- **`ngram_sizes`**: Vector of n-gram sizes being trained (default: `[2, 3]` for bigrams and trigrams)
- **`ngram_weights`**: HashMap mapping n-gram size to its weight (weights are normalized to sum to 1.0)
- **`vocabulary`**: All unique words found in the training data

## Methods

### `new(n: usize) -> Self`

Creates a new `LanguageModel` instance.

**Parameters:**
- `n`: Primary n-gram size (for backward compatibility, typically `3`)

**Default Configuration:**
- N-gram sizes: `[2, 3]` (bigrams and trigrams)
- Weights: Trigram = 0.625, Bigram = 0.375 (normalized)
- Temperature: `1.0` (normal randomness)
- Top-k: `40` (considers top 40 candidates)

**Example:**
```rust
let model = LanguageModel::new(3);
```

---

### `train(&mut self, text: &str)`

Trains the model on the provided text data.

**Parameters:**
- `text`: Training text as a string

**What it does:**
1. Tokenizes the text
2. Builds n-gram statistics for all configured n-gram sizes
3. Computes TF-IDF vectors for semantic search
4. Indexes sentences for pattern matching
5. Creates word-to-context mappings

**Example:**
```rust
let mut model = LanguageModel::new(3);
let training_text = "Mia was getting ready for school. She realized she forgot her homework.";
model.train(&training_text);
```

**Note:** Question words (who, what, where, when, why, how, which, whose, whom) are automatically filtered from training to avoid confusion with query wildcards.

---

### `generate_response(&self, query: &str) -> String`

Generates a response to a query/question.

**Parameters:**
- `query`: The question or query string

**Returns:**
- `String`: The generated response

**Algorithm:**
1. **Direct Pattern Matching**: Searches for exact matches in training text
2. **Wildcard Replacement**: If query contains question words, replaces them with predicted words
3. **Continuation Generation**: Generates the rest of the answer using:
   - Direct sentence continuations (weight: 0.6)
   - Sentence-based tokens (weight: 0.25-0.5)
   - Sentence-filtered n-grams (weight: 0.15-0.5)
   - TF-IDF relevance boosting (1.0-3.5x multiplier)
   - Temperature and top-k sampling

**Example:**
```rust
let response = model.generate_response("who was getting ready");
// Returns: "Mia was getting ready for school."
```

**Question Words (Wildcards):**
- `who`, `what`, `where`, `when`, `why`, `how`, `which`, `whose`, `whom`

These are treated as wildcards that get replaced with predicted words.

---

### `set_temperature(&mut self, temperature: f64)`

Sets the temperature for sampling (controls randomness in generation).

**Parameters:**
- `temperature`: Temperature value (must be > 0.01)
  - `< 1.0`: More deterministic, focused on high-probability tokens
  - `= 1.0`: Normal randomness (default)
  - `> 1.0`: More random, flatter distribution

**Example:**
```rust
let mut model = LanguageModel::new(3);
model.set_temperature(0.8);  // More deterministic
model.set_temperature(1.2);  // More creative/random
```

**Note:** Temperature is applied during candidate selection in `generate_response()`.

---

### `set_top_k(&mut self, top_k: usize)`

Sets the top-k value for sampling (limits candidate pool).

**Parameters:**
- `top_k`: Number of top candidates to consider
  - `0`: No limit, consider all candidates
  - `> 0`: Only consider top-k most likely candidates (reduces low-quality outputs)

**Default:** `40`

**Example:**
```rust
let mut model = LanguageModel::new(3);
model.set_top_k(20);  // Only consider top 20 candidates
model.set_top_k(0);   // Consider all candidates (no limit)
```

**Note:** Top-k filtering is applied after sorting candidates by probability.

---

## Complete Example

```rust
use language_model::LanguageModel;

fn main() {
    // Create model
    let mut model = LanguageModel::new(3);
    
    // Configure sampling
    model.set_temperature(0.9);  // Slightly more deterministic
    model.set_top_k(30);          // Consider top 30 candidates
    
    // Train on data
    let training_data = std::fs::read_to_string("input.txt")
        .expect("Failed to read input.txt");
    model.train(&training_data);
    
    // Generate responses
    println!("{}", model.generate_response("who was getting ready"));
    println!("{}", model.generate_response("what did mia realize"));
    
    // Access model state
    println!("Vocabulary size: {}", model.vocabulary.len());
    println!("N-gram sizes: {:?}", model.ngram_sizes);
    println!("Weights: {:?}", model.ngram_weights);
}
```

## Internal Architecture

### Training Phase

1. **Tokenization**: Splits text into tokens (words with punctuation)
2. **N-gram Counting**: Counts sequences of n words for each n-gram size
3. **TF-IDF Computation**: 
   - Computes Term Frequency (TF) for each word in each sentence
   - Computes Inverse Document Frequency (IDF) for each word
   - Creates TF-IDF vectors: `TF-IDF = TF × IDF`
4. **Sentence Indexing**: Maps words to sentences containing them
5. **Context Windows**: Stores words before/after each word for context matching

### Inference Phase

1. **Query Processing**:
   - Tokenizes query
   - Identifies question words (wildcards)
   
2. **Direct Pattern Matching**:
   - Searches raw training text for exact patterns
   - Returns immediately if match found

3. **Sentence Matching** (if no direct match):
   - **Power Set Matching**: Generates all subsets of query words, finds sentences matching them
   - **TF-IDF Similarity**: Computes cosine similarity between query and sentence vectors
   - Ranks sentences by relevance

4. **Candidate Generation**:
   - Extracts candidates from top matching sentences
   - Uses multi-gram ensemble (weighted combination of bigrams and trigrams)
   - Applies TF-IDF relevance boost (1.0-3.5x multiplier)

5. **Sampling**:
   - Applies Laplace smoothing
   - Sorts by probability
   - Applies top-k filtering
   - Applies temperature scaling
   - Weighted random selection

## Configuration Tips

### For More Deterministic Output
```rust
model.set_temperature(0.5);  // Lower temperature
model.set_top_k(10);         // Fewer candidates
```

### For More Creative Output
```rust
model.set_temperature(1.5);  // Higher temperature
model.set_top_k(100);         // More candidates
```

### For Balanced Output (Default)
```rust
model.set_temperature(1.0);  // Default
model.set_top_k(40);         // Default
```

## Performance Considerations

- **Memory**: Scales with vocabulary size and number of sentences
- **Speed**: 
  - Training: O(n) where n is training text length
  - Inference: O(m) where m is number of sentences (for matching)
- **No GPU Required**: All operations are CPU-based statistical lookups

## Limitations

- Context window is limited to n-gram size (typically 2-3 words)
- Quality depends heavily on training data
- No general knowledge (domain-specific only)
- Question words are filtered from training data

