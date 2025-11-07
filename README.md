# NarrowMind

N-gram language model with multi-gram ensemble and direct pattern matching.

A lightweight question-answering system that learns from training text. Uses statistical n-gram models combined with direct text pattern matching to answer questions with wildcards (who, what, where, etc.).

## Features

- **Multi-gram ensemble**: Weighted combination of bigrams and trigrams
- **Direct pattern matching**: Fast first-layer search for exact answers
- **Sentence-based context**: Power set matching for relevant continuations
- **Question answering**: Wildcard replacement (who, what, where, etc.)

## Technologies & Concepts

- **N-grams & Markov Chains**: Statistical word sequence prediction
- **Multi-gram Ensemble**: Weighted combination of different n-gram sizes
- **Laplace Smoothing**: Probability estimation for unseen sequences
- **Temperature Sampling**: Controls randomness in token selection
- **Top-k Sampling**: Limits candidate pool to most likely tokens
- **Power Set Matching**: Finds sentences matching query word combinations
- **Direct Pattern Matching**: Fast exact text search for answers

## Usage

1. Place training data in `input.txt`
2. Run: `cargo run`
3. Ask questions using question words as wildcards:
   - `who was getting ready`
   - `what did mia realize`
   - `she realized what`

## Architecture

- `src/main.rs` - CLI application
- `src/language_model.rs` - Model implementation

## Requirements

- Rust 1.70+
- Training data in `input.txt`
