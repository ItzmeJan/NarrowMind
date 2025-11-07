# NarrowMind

N-gram language model with multi-gram ensemble and direct pattern matching.

## Features

- **Multi-gram ensemble**: Weighted combination of bigrams and trigrams
- **Direct pattern matching**: Fast first-layer search for exact answers
- **Sentence-based context**: Power set matching for relevant continuations
- **Question answering**: Wildcard replacement (who, what, where, etc.)

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
