# LanguageModel API

## Struct

```rust
pub struct LanguageModel {
    pub ngram_sizes: Vec<usize>,
    pub ngram_weights: HashMap<usize, f64>,
    pub vocabulary: Vec<String>,
}
```

## Methods

### `new(n: usize) -> Self`

Creates a new model instance.

**Defaults:**
- N-grams: `[2, 3]` (bigrams & trigrams)
- Temperature: `1.0`
- Top-k: `40`

```rust
let model = LanguageModel::new(3);
```

---

### `train(&mut self, text: &str)`

Trains the model on text data.

```rust
model.train("Mia was getting ready for school.");
```

---

### `generate_response(&self, query: &str) -> String`

Generates a response to a query.

**Question words (wildcards):** `who`, `what`, `where`, `when`, `why`, `how`, `which`, `whose`, `whom`

```rust
let response = model.generate_response("who was getting ready");
// Returns: "Mia was getting ready for school."
```

---

### `set_temperature(&mut self, temperature: f64)`

Controls randomness in generation.

- `< 1.0`: More deterministic
- `= 1.0`: Normal (default)
- `> 1.0`: More random

```rust
model.set_temperature(0.8);
```

---

### `set_top_k(&mut self, top_k: usize)`

Limits candidate selection to top-k most likely tokens.

- `0`: No limit
- `> 0`: Consider only top-k candidates

```rust
model.set_top_k(20);
```

---

## Example

```rust
let mut model = LanguageModel::new(3);
model.set_temperature(0.9);
model.set_top_k(30);
model.train(&training_data);
let response = model.generate_response("who was getting ready");
```
