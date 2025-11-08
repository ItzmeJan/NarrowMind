# NarrowMind

N-gram language model with multi-gram ensemble and direct pattern matching.

A lightweight question-answering system that learns from training text. Uses statistical n-gram models combined with direct text pattern matching to answer questions with wildcards (who, what, where, etc.).

## Features

- **Multi-gram ensemble**: Weighted combination of bigrams and trigrams
- **Direct pattern matching**: Fast first-layer search for exact answers (#1 preference)
- **TF-IDF vector search**: Cosine similarity for sentence selection (#2 preference)
- **TF-IDF relevance scoring**: Boosts contextually relevant words in continuation generation
- **Sentence-based context**: Power set matching for relevant continuations
- **Question answering**: Wildcard replacement (who, what, where, etc.)

## Technologies & Concepts

- **N-grams & Markov Chains**: Statistical word sequence prediction
- **Multi-gram Ensemble**: Weighted combination of different n-gram sizes
- **Laplace Smoothing**: Probability estimation for unseen sequences
- **Temperature Sampling**: Controls randomness in token selection
- **Top-k Sampling**: Limits candidate pool to most likely tokens
- **Power Set Matching**: Finds sentences matching query word combinations (#1)
- **TF-IDF & Cosine Similarity**: Vector-based sentence ranking (#2 fallback)
- **TF-IDF Relevance Scoring**: Boosts contextually relevant words (1.0-3.5x multiplier)
  - Similarity boost: Up to 2.0x for words in sentences similar to context
  - IDF boost: Up to 1.5x for important/rare words
- **Direct Pattern Matching**: Fast exact text search for answers

## Usage

1. Place training data in `input.txt`
2. Run: `cargo run`
3. Ask questions using question words as wildcards:
   - `who was getting ready`
   - `what did mia realize`
   - `she realized what`

## Algorithm Order

When processing questions, the system uses algorithms in this order:

1. **Direct Pattern Matching** (First Layer)
   - Searches raw training text for exact matches
   - Returns answer phrases immediately if found

2. **Wildcard Replacement** (if direct match fails)
   - **Strategy 1**: Contextual candidates using sentence matching
     - Power set matching (#1 preference) - matches all word subsets
     - TF-IDF vector similarity (#2 preference) - fallback when power set is weak
   - **Strategy 2**: N-gram matching with context
   - Both strategies combined with TF-IDF relevance scoring

3. **Continuation Generation** (after wildcards filled)
   - **Direct sentence continuations** (weight: 0.6) - extracts next words from matching sentences
   - **Sentence-based tokens** (weight: 0.25-0.5) - tokens from top matching sentences
   - **Sentence-filtered n-grams** (weight: 0.15-0.5) - n-grams filtered by sentence relevance
   - **Fallback**: Unfiltered multi-gram ensemble (trigrams + bigrams)
   - **Final fallback**: Unigrams (most frequent words)
   - All candidates boosted by TF-IDF relevance scoring (1.0-3.5x multiplier)

## Architecture

- `src/main.rs` - CLI application
- `src/language_model.rs` - Model implementation

## Requirements

- Rust 1.70+
- Training data in `input.txt`
