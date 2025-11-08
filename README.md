# NarrowMind

**A lightweight statistical language model for question-answering and text generation.**

NarrowMind is a small language model that learns patterns from training text to answer questions and generate contextual continuations. While it's not a transformer-based neural network like GPT, NarrowMind combines **statistical n-gram modeling** with modern language modeling techniques (temperature sampling, top-k sampling, TF-IDF) to provide fast, memory-efficient language understanding.

## What is NarrowMind?

NarrowMind is a **hybrid statistical language model** that combines traditional n-gram techniques with modern language modeling concepts. While it's not a neural network or transformer model like GPT, it incorporates several techniques used in modern LLMs:

- **Temperature Sampling**: Controls randomness in text generation (same concept as GPT)
- **Top-k Sampling**: Limits predictions to most likely candidates (used in GPT-2/3)
- **TF-IDF Vectors**: Semantic representations for finding similar content
- **Statistical N-grams**: Traditional Markov chain-based word prediction

It learns from your training data by:
- Analyzing word sequences and their probabilities
- Building semantic representations using TF-IDF vectors
- Matching patterns in training text to answer questions
- Generating contextually relevant text continuations

**Note**: This is NOT a GPT or generative pretrained transformer model. It's a statistical approach using n-grams, Markov chains, and information retrieval techniques, but it incorporates modern sampling strategies (temperature, top-k) and semantic search (TF-IDF) inspired by modern language models. This allows it to perform similar tasks like question-answering and text generation with much lower computational requirements.

## Capabilities

- **Question Answering**: Understands questions with wildcards (who, what, where, when, why, how)
- **Text Generation**: Generates contextually relevant continuations
- **Semantic Understanding**: Uses TF-IDF vectors to find semantically similar content
- **Pattern Matching**: Direct text pattern matching for fast, accurate answers
- **Context-Aware**: Considers surrounding words and sentence structure

## How It Works

NarrowMind uses a multi-layered approach to understand and generate text:

### 1. **Direct Pattern Matching** (Fast Path)
   - Searches training text for exact question-answer patterns
   - Returns immediate answers when patterns match
   - Highest priority for speed and accuracy

### 2. **Semantic Sentence Matching**
   - **Power Set Matching**: Finds sentences matching all combinations of query words
   - **TF-IDF Vector Search**: Uses cosine similarity to find semantically similar sentences
   - Ranks sentences by relevance to the query

### 3. **Contextual Word Prediction**
   - **Multi-gram Ensemble**: Combines bigrams and trigrams with weighted probabilities
   - **TF-IDF Relevance Scoring**: Boosts contextually relevant words (1.0-3.5x multiplier)
     - Words in similar sentences get up to 2.0x boost
     - Important/rare words get up to 1.5x boost
   - **Adaptive Weighting**: Dynamically adjusts based on available context

### 4. **Text Generation Pipeline**
   - Extracts next words from matching sentences (most natural)
   - Filters n-grams by sentence relevance
   - Falls back to statistical n-gram predictions
   - Uses temperature and top-k sampling for controlled randomness

## Technical Architecture

### Core Technologies

**Statistical Foundations:**
- **N-gram Language Modeling**: Statistical word sequence prediction using Markov chains
- **Multi-gram Ensemble**: Weighted combination of bigrams (n=2) and trigrams (n=3)
- **Laplace Smoothing**: Handles unseen word sequences gracefully
- **Power Set Matching**: Finds sentences matching all word subset combinations

**Modern LLM Techniques (Inspired by GPT):**
- **Temperature Sampling**: Controls randomness in generation (same as GPT - lower = more deterministic, higher = more creative)
- **Top-k Sampling**: Limits predictions to top-k most likely candidates (used in GPT-2/3 for quality control)
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Semantic vector representations for finding relevant content
- **Cosine Similarity**: Measures semantic similarity between text vectors
- **TF-IDF Relevance Scoring**: Boosts contextually relevant words (similar to attention mechanisms in transformers)

### Model Architecture

```
Training Data
    ↓
[Tokenization & Preprocessing]
    ↓
[Multi-gram Statistics] ──→ [TF-IDF Vectors] ──→ [Sentence Index]
    ↓                              ↓                      ↓
[Context Maps]              [Semantic Search]    [Pattern Matching]
    ↓                              ↓                      ↓
                    [Question Processing Pipeline]
                              ↓
                    [Answer Generation]
```

## Usage

### Quick Start

1. **Prepare training data**: Place your text in `input.txt`
2. **Train the model**: Run `cargo run`
3. **Ask questions**: Use question words as wildcards
   ```
   > who was getting ready
   > what did mia realize
   > she realized what
   ```

### Example Interactions

```
Training: "Mia was getting ready for school. She realized she forgot her homework."

Query: "who was getting ready"
Response: "Mia was getting ready for school."

Query: "what did mia realize"
Response: "Mia realized she forgot her homework."
```

## Performance Characteristics

- **Memory Efficient**: No large neural network weights
- **Fast Inference**: Statistical lookups, no GPU required
- **Interpretable**: Can trace why specific answers were chosen
- **Lightweight**: Suitable for embedded systems and low-resource environments

## Limitations

- **Not a Neural Network**: This is a statistical model, not a transformer-based neural network like GPT
- **Training Data Dependent**: Quality depends on training text (no pretraining on massive datasets)
- **Limited Context Window**: Uses n-gram context (typically 2-3 words) vs. thousands of tokens in GPT
- **Domain-Specific**: Must train on your specific domain (no general knowledge like GPT)
- **Simpler Representations**: Uses TF-IDF vectors instead of learned embeddings

## What It Shares with GPT

Despite being a statistical model, NarrowMind incorporates several techniques popularized by modern language models:

- ✅ **Temperature Sampling**: Same randomness control mechanism as GPT
- ✅ **Top-k Sampling**: Same candidate filtering approach used in GPT-2/3
- ✅ **Semantic Search**: TF-IDF vectors (simpler alternative to transformer embeddings)
- ✅ **Context-Aware Generation**: Considers surrounding context for predictions
- ✅ **Probabilistic Text Generation**: Uses probability distributions for word selection

## Project Structure

```
NarrowMind/
├── src/
│   ├── main.rs              # CLI interface
│   └── language_model.rs    # Core model implementation
├── input.txt                # Training data
└── README.md                # This file
```

## Requirements

- **Rust**: 1.70 or later
- **Training Data**: Text file (`input.txt`) with your domain content
- **No External Dependencies**: Uses only Rust standard library and `rand` crate

## How It Compares to GPT

| Feature | NarrowMind | GPT/Transformers |
|---------|-----------|------------------|
| **Architecture** | Statistical n-grams | Neural network (transformer) |
| **Training** | Counts word sequences | Gradient descent on billions of parameters |
| **Memory** | ~MBs | ~GBs to TBs |
| **Speed** | Instant (lookups) | Slower (matrix operations) |
| **Context Window** | 2-3 words (n-grams) | Thousands of tokens |
| **Pretraining** | No | Yes (on massive datasets) |
| **GPU Required** | No | Yes (for training) |
| **Temperature Sampling** | ✅ Yes (same concept) | ✅ Yes |
| **Top-k Sampling** | ✅ Yes (same concept) | ✅ Yes |
| **Semantic Search** | ✅ TF-IDF vectors | ✅ Embeddings |
| **Interpretability** | ✅ High (statistical) | ⚠️ Lower (black box) |

---

**NarrowMind**: Think small, understand deeply. 🧠
