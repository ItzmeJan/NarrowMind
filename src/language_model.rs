use std::collections::HashMap;
use rand::Rng;

type NGram = Vec<String>;
type NGramCount = HashMap<NGram, u32>;
type NGramContext = HashMap<Vec<String>, Vec<(String, u32)>>;

// Context entry: stores a sentence/context and its tokens
#[derive(Clone)]
struct ContextEntry {
    tokens: Vec<String>,
    text: String, // Original text for reference
}

pub struct LanguageModel {
    n: usize, // Primary n-gram size (for backward compatibility)
    pub ngram_sizes: Vec<usize>, // All n-gram sizes to train (e.g., [2, 3, 4])
    pub ngram_weights: HashMap<usize, f64>, // Weights for each n-gram size (must sum to ~1.0)
    ngram_counts: NGramCount, // Combined counts (for backward compatibility)
    ngram_contexts: NGramContext, // Primary n-gram contexts (for backward compatibility)
    multi_ngram_contexts: HashMap<usize, NGramContext>, // Separate contexts for each n-gram size
    pub vocabulary: Vec<String>,
    unigram_counts: HashMap<String, u32>, // For smoothing and backoff
    total_unigrams: u32, // Total word count for probability calculations
    temperature: f64, // Controls randomness: 1.0 = normal, <1.0 = more deterministic, >1.0 = more random
    top_k: usize, // Only consider top-k most likely tokens (0 = no limit)
    // Full text context scanning
    contexts: Vec<ContextEntry>, // All sentence-level contexts from training data
    word_to_contexts: HashMap<String, Vec<usize>>, // Maps words to context indices where they appear
    context_windows: HashMap<String, Vec<(Vec<String>, Vec<String>)>>, // Word -> (before_context, after_context) pairs
    raw_training_text: String, // Raw training text for direct pattern matching
    // TF-IDF for vector-based sentence selection
    tfidf_vectors: Vec<HashMap<String, f64>>, // TF-IDF vectors for each sentence
    idf_scores: HashMap<String, f64>, // Inverse document frequency for each word
    total_sentences: usize, // Total number of sentences for IDF calculation
}

impl LanguageModel {
    pub fn new(n: usize) -> Self {
        // Default: train bigrams (2) and trigrams (3)
        // Weights: trigram=0.5, bigram=0.3, 4-gram=0.2 (if used)
        let ngram_sizes = vec![2, 3]; // Bigrams and trigrams
        let mut ngram_weights = HashMap::new();
        ngram_weights.insert(3, 0.5); // Trigram weight
        ngram_weights.insert(2, 0.3); // Bigram weight
        // Normalize weights
        let total_weight: f64 = ngram_weights.values().sum();
        for weight in ngram_weights.values_mut() {
            *weight /= total_weight;
        }
        
        let mut multi_ngram_contexts = HashMap::new();
        for &size in &ngram_sizes {
            multi_ngram_contexts.insert(size, HashMap::new());
        }
        
        Self {
            n,
            ngram_sizes,
            ngram_weights,
            ngram_counts: HashMap::new(),
            ngram_contexts: HashMap::new(),
            multi_ngram_contexts,
            vocabulary: Vec::new(),
            unigram_counts: HashMap::new(),
            total_unigrams: 0,
            temperature: 1.0, // Default: normal randomness
            top_k: 40, // Default: consider top 40 candidates
            contexts: Vec::new(),
            word_to_contexts: HashMap::new(),
            context_windows: HashMap::new(),
            raw_training_text: String::new(),
            tfidf_vectors: Vec::new(),
            idf_scores: HashMap::new(),
            total_sentences: 0,
        }
    }
    
    /// Create model with custom n-gram sizes and weights
    fn new_with_ngrams(ngram_sizes: Vec<usize>, weights: Vec<f64>) -> Self {
        if ngram_sizes.len() != weights.len() {
            panic!("ngram_sizes and weights must have the same length");
        }
        
        let primary_n = *ngram_sizes.iter().max().unwrap_or(&3);
        let mut ngram_weights = HashMap::new();
        for (size, weight) in ngram_sizes.iter().zip(weights.iter()) {
            ngram_weights.insert(*size, *weight);
        }
        
        // Normalize weights to sum to 1.0
        let total_weight: f64 = ngram_weights.values().sum();
        if total_weight > 0.0 {
            for weight in ngram_weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        let mut multi_ngram_contexts = HashMap::new();
        for &size in &ngram_sizes {
            multi_ngram_contexts.insert(size, HashMap::new());
        }
        
        Self {
            n: primary_n,
            ngram_sizes,
            ngram_weights,
            ngram_counts: HashMap::new(),
            ngram_contexts: HashMap::new(),
            multi_ngram_contexts,
            vocabulary: Vec::new(),
            unigram_counts: HashMap::new(),
            total_unigrams: 0,
            temperature: 1.0,
            top_k: 40,
            contexts: Vec::new(),
            word_to_contexts: HashMap::new(),
            context_windows: HashMap::new(),
            raw_training_text: String::new(),
            tfidf_vectors: Vec::new(),
            idf_scores: HashMap::new(),
            total_sentences: 0,
        }
    }

    pub fn train(&mut self, text: &str) {
        // Store raw training text for direct pattern matching
        self.raw_training_text = text.to_string();
        
        // FULL TEXT SCAN: Extract sentence-level contexts
        // Split text into sentences and store full contexts
        let sentences: Vec<&str> = text.split(|c: char| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        
        // Process each sentence for full context storage
        for sentence in &sentences {
            let sentence_tokens = self.tokenize(sentence);
            if sentence_tokens.is_empty() {
                continue;
            }
            
            // Store the full context
            let context_idx = self.contexts.len();
            self.contexts.push(ContextEntry {
                tokens: sentence_tokens.clone(),
                text: sentence.to_string(),
            });
            
            // Map each word to contexts where it appears
            for (word_pos, token) in sentence_tokens.iter().enumerate() {
                let word = self.extract_word(token).to_lowercase();
                self.word_to_contexts
                    .entry(word.clone())
                    .or_insert_with(Vec::new)
                    .push(context_idx);
                
                // Store context windows (words before and after each word)
                let before: Vec<String> = if word_pos > 0 {
                    let start = word_pos.saturating_sub(5).max(0);
                    sentence_tokens[start..word_pos].to_vec()
                } else {
                    Vec::new()
                };
                
                let after: Vec<String> = if word_pos + 1 < sentence_tokens.len() {
                    let end = (word_pos + 1 + 5).min(sentence_tokens.len());
                    sentence_tokens[word_pos + 1..end].to_vec()
                } else {
                    Vec::new()
                };
                
                self.context_windows
                    .entry(word.clone())
                    .or_insert_with(Vec::new)
                    .push((before, after));
            }
        }
        
        // Tokenize training data and filter out question words
        // Question words are automatically removed to avoid confusion with query wildcards
        let all_tokens = self.tokenize(text);
        
        // Filter out question words from training tokens
        let tokens: Vec<String> = all_tokens.iter()
            .filter(|token| {
                let word = self.extract_word(token);
                !self.is_question_word(word)
            })
            .cloned()
            .collect();
        
        // Train n-grams for all sizes
        for &ngram_size in &self.ngram_sizes {
            if tokens.len() < ngram_size {
                continue;
            }

            // Get the context map for this n-gram size
            let ngram_contexts = self.multi_ngram_contexts.get_mut(&ngram_size).unwrap();

            for i in 0..=tokens.len().saturating_sub(ngram_size) {
                let ngram: NGram = tokens[i..i + ngram_size].to_vec();
                
                // Count n-grams (store in primary ngram_counts for backward compatibility)
                if ngram_size == self.n {
                    *self.ngram_counts.entry(ngram.clone()).or_insert(0) += 1;
                }

                // Build context for generation (context is n-1 tokens, next token is the continuation)
                if i + ngram_size < tokens.len() {
                    let context: Vec<String> = ngram[..ngram_size - 1].to_vec();
                    let next_token = tokens[i + ngram_size].clone();
                    
                    let continuations = ngram_contexts
                        .entry(context)
                        .or_insert_with(Vec::new);
                    
                    // Check if this continuation already exists
                    if let Some(existing) = continuations.iter_mut().find(|(token, _)| token == &next_token) {
                        existing.1 += 1;
                    } else {
                        continuations.push((next_token, 1));
                    }
                }
            }
        }
        
        // Also maintain primary n-gram contexts for backward compatibility
        if let Some(primary_contexts) = self.multi_ngram_contexts.get(&self.n) {
            self.ngram_contexts = primary_contexts.clone();
        }

        // Build vocabulary and unigram counts (only non-question words)
        for token in &tokens {
            if !self.vocabulary.contains(token) {
                self.vocabulary.push(token.clone());
            }
            // Count unigrams for smoothing and backoff
            *self.unigram_counts.entry(token.clone()).or_insert(0) += 1;
            self.total_unigrams += 1;
        }
        
        // Compute TF-IDF vectors for sentence selection (#2 preference)
        self.compute_tfidf();
    }
    
    /// Compute TF-IDF vectors for all sentences
    /// Used for vector-based sentence selection as fallback to power set matching
    fn compute_tfidf(&mut self) {
        self.total_sentences = self.contexts.len();
        if self.total_sentences == 0 {
            return;
        }
        
        // Step 1: Count document frequency (DF) - how many sentences contain each word
        let mut document_frequency: HashMap<String, usize> = HashMap::new();
        
        for context in &self.contexts {
            let mut sentence_words: std::collections::HashSet<String> = std::collections::HashSet::new();
            for token in &context.tokens {
                let word = self.extract_word(token).to_lowercase();
                if !self.is_question_word(&word) {
                    sentence_words.insert(word);
                }
            }
            for word in sentence_words {
                *document_frequency.entry(word).or_insert(0) += 1;
            }
        }
        
        // Step 2: Compute IDF (Inverse Document Frequency)
        // IDF = log(total_sentences / (1 + document_frequency))
        for (word, df) in &document_frequency {
            let idf = ((self.total_sentences as f64) / (1.0 + *df as f64)).ln();
            self.idf_scores.insert(word.clone(), idf);
        }
        
        // Step 3: Compute TF-IDF vectors for each sentence
        self.tfidf_vectors.clear();
        for context in &self.contexts {
            let mut tfidf_vector: HashMap<String, f64> = HashMap::new();
            
            // Count term frequency (TF) in this sentence
            let mut term_frequency: HashMap<String, usize> = HashMap::new();
            let mut total_words = 0;
            
            for token in &context.tokens {
                let word = self.extract_word(token).to_lowercase();
                if !self.is_question_word(&word) {
                    *term_frequency.entry(word.clone()).or_insert(0) += 1;
                    total_words += 1;
                }
            }
            
            // Compute TF-IDF = TF * IDF
            // TF = count(word) / total_words_in_sentence
            for (word, count) in term_frequency {
                let tf = count as f64 / total_words as f64;
                let idf = *self.idf_scores.get(&word).unwrap_or(&0.0);
                tfidf_vector.insert(word, tf * idf);
            }
            
            self.tfidf_vectors.push(tfidf_vector);
        }
    }
    
    /// Find similar contexts using TF-IDF cosine similarity (#2 preference)
    /// Falls back to this when power set matching (#1) doesn't find good matches
    fn find_similar_contexts_tfidf(&self, query_words: &[String]) -> Vec<(usize, f64)> {
        if self.tfidf_vectors.is_empty() || query_words.is_empty() {
            return Vec::new();
        }
        
        // Build query TF-IDF vector
        let mut query_vector: HashMap<String, f64> = HashMap::new();
        let mut query_tf: HashMap<String, usize> = HashMap::new();
        let query_word_count = query_words.len();
        
        for word in query_words {
            let normalized_word = self.extract_word(word).to_lowercase();
            if !self.is_question_word(&normalized_word) {
                *query_tf.entry(normalized_word).or_insert(0) += 1;
            }
        }
        
        // Compute TF-IDF for query
        for (word, count) in query_tf {
            let tf = count as f64 / query_word_count as f64;
            let idf = *self.idf_scores.get(&word).unwrap_or(&0.0);
            query_vector.insert(word, tf * idf);
        }
        
        // Compute cosine similarity with each sentence vector
        let mut similarities: Vec<(usize, f64)> = Vec::new();
        
        for (idx, sentence_vector) in self.tfidf_vectors.iter().enumerate() {
            let similarity = self.cosine_similarity(&query_vector, sentence_vector);
            if similarity > 0.0 {
                similarities.push((idx, similarity));
            }
        }
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Return top matches
        similarities.into_iter().take(30).collect()
    }
    
    /// Compute cosine similarity between two TF-IDF vectors
    fn cosine_similarity(&self, vec1: &HashMap<String, f64>, vec2: &HashMap<String, f64>) -> f64 {
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        
        // Compute dot product and norms
        let all_words: std::collections::HashSet<&String> = vec1.keys().chain(vec2.keys()).collect();
        
        for word in all_words {
            let val1 = vec1.get(word).copied().unwrap_or(0.0);
            let val2 = vec2.get(word).copied().unwrap_or(0.0);
            
            dot_product += val1 * val2;
            norm1 += val1 * val1;
            norm2 += val2 * val2;
        }
        
        // Cosine similarity = dot_product / (sqrt(norm1) * sqrt(norm2))
        let denominator = (norm1.sqrt() * norm2.sqrt()).max(1e-10); // Avoid division by zero
        dot_product / denominator
    }
    
    /// Compute TF-IDF relevance score for a candidate word given the context
    /// Returns a multiplier (>= 1.0) that boosts words more relevant to the context
    fn compute_tfidf_relevance(&self, candidate_word: &str, context_words: &[String]) -> f64 {
        if context_words.is_empty() || self.tfidf_vectors.is_empty() {
            return 1.0; // No boost if no context or no TF-IDF data
        }
        
        let candidate_word_lower = self.extract_word(candidate_word).to_lowercase();
        
        // Build TF-IDF vector for the context
        let mut context_vector: HashMap<String, f64> = HashMap::new();
        let mut context_tf: HashMap<String, usize> = HashMap::new();
        let context_word_count = context_words.len();
        
        for word in context_words {
            let normalized_word = self.extract_word(word).to_lowercase();
            if !self.is_question_word(&normalized_word) {
                *context_tf.entry(normalized_word).or_insert(0) += 1;
            }
        }
        
        // Compute TF-IDF for context
        for (word, count) in context_tf {
            let tf = count as f64 / context_word_count as f64;
            let idf = *self.idf_scores.get(&word).unwrap_or(&0.0);
            context_vector.insert(word, tf * idf);
        }
        
        // Find sentences containing the candidate word and compute average similarity
        let mut total_similarity = 0.0;
        let mut sentence_count = 0;
        
        // Get IDF score of candidate word (how important/rare it is)
        let candidate_idf = *self.idf_scores.get(&candidate_word_lower).unwrap_or(&0.0);
        
        // Find sentences that contain the candidate word
        if let Some(sentence_indices) = self.word_to_contexts.get(&candidate_word_lower) {
            for &sentence_idx in sentence_indices.iter().take(10) { // Limit to top 10 for efficiency
                if let Some(sentence_vector) = self.tfidf_vectors.get(sentence_idx) {
                    let similarity = self.cosine_similarity(&context_vector, sentence_vector);
                    if similarity > 0.0 {
                        total_similarity += similarity;
                        sentence_count += 1;
                    }
                }
            }
        }
        
        // Compute relevance score:
        // 1. Average similarity with sentences containing the candidate (0-1 range)
        // 2. IDF score of candidate (normalized, 0-1 range)
        // Combine them for a relevance multiplier
        
        let avg_similarity = if sentence_count > 0 {
            total_similarity / sentence_count as f64
        } else {
            0.0
        };
        
        // Normalize IDF (assuming max IDF is around 5-6 for rare words)
        // Use log scale: normalize to 0-1 range
        let normalized_idf = (candidate_idf / 6.0).min(1.0).max(0.0);
        
        // Relevance multiplier: 
        // - Base: 1.0 (no boost)
        // - Similarity boost: up to 2.0x for high similarity
        // - IDF boost: up to 1.5x for important/rare words
        // Combined: 1.0 to ~3.5x multiplier
        let similarity_boost = 1.0 + avg_similarity; // 1.0 to 2.0
        let idf_boost = 1.0 + (normalized_idf * 0.5); // 1.0 to 1.5
        
        similarity_boost * idf_boost
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let text_lower = text.to_lowercase();
        let mut current_word = String::new();
        
        for ch in text_lower.chars() {
            if ch.is_whitespace() {
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
            } else if ch.is_alphanumeric() || ch == '\'' {
                // Include apostrophes for contractions
                current_word.push(ch);
            } else {
                // Punctuation encountered - attach to current word if it exists
                if !current_word.is_empty() {
                    current_word.push(ch);
                    tokens.push(current_word.clone());
                    current_word.clear();
                } else {
                    // Punctuation at start (shouldn't happen often, but handle it)
                    tokens.push(ch.to_string());
                }
            }
        }
        
        // Add any remaining word (without trailing punctuation)
        if !current_word.is_empty() {
            tokens.push(current_word);
        }
        
        tokens
    }

    fn is_sentence_ender(&self, token: &str) -> bool {
        token.ends_with('.') || token.ends_with('!') || token.ends_with('?')
    }

    fn is_pause(&self, token: &str) -> bool {
        token.ends_with(',') || token.ends_with(';') || token.ends_with(':')
    }

    fn extract_word<'a>(&self, token: &'a str) -> &'a str {
        // Remove trailing punctuation to get the base word
        token.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '\'')
    }

    /// Extract next words directly from matching sentences based on context position
    /// This provides more natural continuations following real sentence patterns
    /// IMPROVED: Handles partial matches and extracts multiple next words for better flow
    fn get_direct_sentence_continuations(&self, context: &[String], query_words: &[String]) -> Vec<(String, u32)> {
        let mut continuations: HashMap<String, u32> = HashMap::new();
        
        // Find top matching sentences
        let top_sentences = self.find_similar_contexts(query_words);
        
        // Normalize context for matching
        let context_words_normalized: Vec<String> = context.iter()
            .map(|t| self.extract_word(t).to_lowercase())
            .collect();
        
        if context_words_normalized.is_empty() {
            return Vec::new();
        }
        
        for (sentence_idx, match_score) in top_sentences.iter().take(20) {
            if let Some(context_entry) = self.contexts.get(*sentence_idx) {
                let sentence_words: Vec<String> = context_entry.tokens.iter()
                    .map(|t| self.extract_word(t).to_lowercase())
                    .collect();
                
                // Find where our context appears in this sentence (exact match or suffix match)
                for start_idx in 0..sentence_words.len() {
                    // Try exact match first
                    let mut matches_exact = true;
                    let mut matched_length = 0;
                    
                    for (i, ctx_word) in context_words_normalized.iter().enumerate() {
                        if start_idx + i >= sentence_words.len() {
                            matches_exact = false;
                            break;
                        }
                        if sentence_words[start_idx + i] == *ctx_word {
                            matched_length += 1;
                        } else {
                            matches_exact = false;
                            break;
                        }
                    }
                    
                    // If we have a match (exact or partial), extract next word(s)
                    if matches_exact && matched_length > 0 {
                        let next_pos = start_idx + matched_length;
                        
                        // Extract next 1-2 words for better context
                        for offset in 0..2 {
                        if next_pos + offset < context_entry.tokens.len() {
                            let next_token = &context_entry.tokens[next_pos + offset];
                            let _word = self.extract_word(next_token).to_lowercase();
                                
                                // Skip question words and words already in context
                                if !self.is_question_word(&_word) && !context_words_normalized.contains(&_word) {
                                    // Weight by: match score, how much of context matched, position in sentence
                                    let match_ratio = matched_length as f64 / context_words_normalized.len() as f64;
                                    let position_bonus = if start_idx < 5 { 1.5 } else { 1.0 }; // Earlier in sentence = better
                                    let offset_penalty = if offset == 0 { 1.0 } else { 0.7 }; // First word after = better
                                    
                                    let weight = ((*match_score * match_ratio * position_bonus * offset_penalty) as u32) * 5;
                                    *continuations.entry(next_token.clone()).or_insert(0) += weight;
                                }
                            }
                        }
                    }
                    
                    // Also try suffix matching: if context ends match sentence
                    if context_words_normalized.len() > 1 && start_idx + context_words_normalized.len() <= sentence_words.len() {
                        let context_suffix = &context_words_normalized[context_words_normalized.len().saturating_sub(2)..];
                        let sentence_window = &sentence_words[start_idx..(start_idx + context_suffix.len()).min(sentence_words.len())];
                        
                        if context_suffix == sentence_window {
                            let next_pos = start_idx + context_suffix.len();
                            if next_pos < context_entry.tokens.len() {
                                let next_token = &context_entry.tokens[next_pos];
                                let _word = self.extract_word(next_token).to_lowercase();
                                
                                if !self.is_question_word(&_word) && !context_words_normalized.contains(&_word) {
                                    // Lower weight for suffix matches
                                    let weight = ((*match_score * 0.7) as u32) * 3;
                                    *continuations.entry(next_token.clone()).or_insert(0) += weight;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        continuations.into_iter().collect()
    }

    /// Get weighted candidates from all n-gram sizes
    /// Combines predictions from bigrams, trigrams, etc. with their respective weights
    fn get_multi_ngram_candidates(&self, context: &[String], query_words: &[String], stop_at_sentence_end: bool) -> Vec<(String, f64)> {
        let mut combined_candidates: HashMap<String, f64> = HashMap::new();
        
        // Get candidates from each n-gram size
        for &ngram_size in &self.ngram_sizes {
            let weight = *self.ngram_weights.get(&ngram_size).unwrap_or(&0.0);
            if weight == 0.0 {
                continue;
            }
            
            // Get candidates for this n-gram size
            let candidates = self.get_ngram_candidates_for_size(ngram_size, context, query_words, stop_at_sentence_end);
            
            // Combine with weighted scores
            for (token, count) in candidates {
                *combined_candidates.entry(token).or_insert(0.0) += (count as f64) * weight;
            }
        }
        
        combined_candidates.into_iter().collect()
    }
    
    /// Get n-gram candidates for a specific n-gram size
    fn get_ngram_candidates_for_size(&self, ngram_size: usize, context: &[String], query_words: &[String], stop_at_sentence_end: bool) -> Vec<(String, u32)> {
        let mut candidates: HashMap<String, u32> = HashMap::new();
        
        // Get the n-gram contexts for this size
        let ngram_contexts = match self.multi_ngram_contexts.get(&ngram_size) {
            Some(ctx) => ctx,
            None => return Vec::new(),
        };
        
        // Strategy 1: Try full n-gram context (n-1 tokens)
        if context.len() >= ngram_size - 1 {
            let context_key: Vec<String> = context[context.len().saturating_sub(ngram_size - 1)..].to_vec();
            
            if let Some(continuations) = ngram_contexts.get(&context_key) {
                for (token, count) in continuations {
                    *candidates.entry(token.clone()).or_insert(0) += count;
                }
            }
        }
        
        // Strategy 2: Backoff to (n-2) context (if n > 2)
        if candidates.is_empty() && ngram_size > 2 && context.len() >= ngram_size - 2 {
            let context_key: Vec<String> = context[context.len().saturating_sub(ngram_size - 2)..].to_vec();
            
            for (ctx_key, continuations) in ngram_contexts {
                if ctx_key.len() >= context_key.len() {
                    let ctx_suffix = &ctx_key[ctx_key.len() - context_key.len()..];
                    if ctx_suffix == context_key.as_slice() {
                        for (token, count) in continuations {
                            *candidates.entry(token.clone()).or_insert(0) += count;
                        }
                    }
                }
            }
        }
        
        // Strategy 3: Backoff to bigram (last token only)
        if candidates.is_empty() && context.len() >= 1 {
            let last_token = &context[context.len() - 1];
            
            for (ctx_key, continuations) in ngram_contexts {
                if !ctx_key.is_empty() && ctx_key[ctx_key.len() - 1] == *last_token {
                    for (token, count) in continuations {
                        *candidates.entry(token.clone()).or_insert(0) += count;
                    }
                }
            }
        }
        
        candidates.into_iter().collect()
    }

    /// Get n-gram candidates with sentence-guided filtering
    /// Only includes n-grams that appear in top matching sentences
    /// NOW USES MULTI-GRAM ENSEMBLE: combines all n-gram sizes with weights
    fn get_sentence_filtered_ngrams(&self, context: &[String], query_words: &[String], stop_at_sentence_end: bool) -> Vec<(String, u32)> {
        let mut ngram_candidates: HashMap<String, u32> = HashMap::new();
        
        // Get top matching sentences to use as filter and for weighting
        let top_sentences = self.find_similar_contexts(query_words);
        let mut sentence_words_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut word_to_sentence_scores: HashMap<String, f64> = HashMap::new();
        
        // Collect all words from top matching sentences with their sentence scores
        for (sentence_idx, sentence_score) in top_sentences.iter().take(15) {
            if let Some(context_entry) = self.contexts.get(*sentence_idx) {
                for token in &context_entry.tokens {
                    let word = self.extract_word(token).to_lowercase();
                    sentence_words_set.insert(word.clone());
                    // Track which sentences contain this word and their scores
                    let current_max = word_to_sentence_scores.get(&word).copied().unwrap_or(0.0);
                    word_to_sentence_scores.insert(word, current_max.max(*sentence_score));
                }
            }
        }
        
        // MULTI-GRAM ENSEMBLE: Get candidates from all n-gram sizes with weighted combination
        for &ngram_size in &self.ngram_sizes {
            let ngram_weight = *self.ngram_weights.get(&ngram_size).unwrap_or(&0.0);
            if ngram_weight == 0.0 {
                continue;
            }
            
            // Get the n-gram contexts for this size
            let ngram_contexts = match self.multi_ngram_contexts.get(&ngram_size) {
                Some(ctx) => ctx,
                None => continue,
            };
            
            // Strategy 1: Full n-gram context
            if context.len() >= ngram_size - 1 {
                let context_key: Vec<String> = context[context.len().saturating_sub(ngram_size - 1)..].to_vec();
                
                if let Some(continuations) = ngram_contexts.get(&context_key) {
                    for (token, count) in continuations {
                        let word = self.extract_word(token).to_lowercase();
                        // Filter: only include if word appears in top matching sentences
                        if sentence_words_set.contains(&word) {
                            // Get the highest sentence score for this word
                            let sentence_boost = word_to_sentence_scores.get(&word).copied().unwrap_or(0.0);
                            
                            // Combine: n-gram weight * sentence boost * count
                            let boosted_count = (*count as f64) * (1.0 + sentence_boost * 0.1) * ngram_weight;
                            
                            *ngram_candidates.entry(token.clone()).or_insert(0) += boosted_count as u32;
                        }
                    }
                }
            }
            
            // Strategy 2: Backoff to (n-2) context
            if ngram_candidates.is_empty() && ngram_size > 2 && context.len() >= ngram_size - 2 {
                let context_key: Vec<String> = context[context.len().saturating_sub(ngram_size - 2)..].to_vec();
                
                for (ctx_key, continuations) in ngram_contexts {
                    if ctx_key.len() >= context_key.len() {
                        let ctx_suffix = &ctx_key[ctx_key.len() - context_key.len()..];
                        if ctx_suffix == context_key.as_slice() {
                            for (token, count) in continuations {
                                let word = self.extract_word(token).to_lowercase();
                                if sentence_words_set.contains(&word) {
                                    let sentence_boost = word_to_sentence_scores.get(&word).copied().unwrap_or(0.0);
                                    let boosted_count = (*count as f64) * (1.0 + sentence_boost * 0.1) * ngram_weight;
                                    *ngram_candidates.entry(token.clone()).or_insert(0) += boosted_count as u32;
                                }
                            }
                        }
                    }
                }
            }
            
            // Strategy 3: Bigram backoff
            if ngram_candidates.is_empty() && context.len() >= 1 {
                let last_token = &context[context.len() - 1];
                
                for (ctx_key, continuations) in ngram_contexts {
                    if !ctx_key.is_empty() && ctx_key[ctx_key.len() - 1] == *last_token {
                        for (token, count) in continuations {
                            let word = self.extract_word(token).to_lowercase();
                            if sentence_words_set.contains(&word) {
                                let sentence_boost = word_to_sentence_scores.get(&word).copied().unwrap_or(0.0);
                                let boosted_count = (*count as f64) * (1.0 + sentence_boost * 0.1) * ngram_weight;
                                *ngram_candidates.entry(token.clone()).or_insert(0) += boosted_count as u32;
                            }
                        }
                    }
                }
            }
        }
        
        ngram_candidates.into_iter().collect()
    }

    fn generate_continuation(&self, context: &[String], stop_at_sentence_end: bool) -> Option<String> {
        if context.is_empty() {
            return None;
        }

        // HYBRID APPROACH: Combine sentence-based, direct continuations, and n-gram candidates
        let context_words: Vec<String> = context.iter()
            .map(|t| self.extract_word(t).to_lowercase())
            .collect();
        
        // Get three types of candidates:
        // 1. Sentence-based tokens (general matching)
        let sentence_based_tokens = self.get_tokens_from_matching_sentences(&context_words);
        
        // 2. Direct sentence continuations (extract next word from matching sentences)
        let direct_continuations = self.get_direct_sentence_continuations(&context, &context_words);
        
        // 3. N-gram candidates (sentence-filtered for better context relevance)
        let ngram_candidates = self.get_sentence_filtered_ngrams(&context, &context_words, stop_at_sentence_end);
        
        // ADAPTIVE WEIGHTED COMBINATION: Dynamically adjust weights based on quality of matches
        let mut combined_candidates: HashMap<String, f64> = HashMap::new();
        
        // Calculate adaptive weights based on how many candidates we have from each source
        let direct_count = direct_continuations.len();
        let sentence_count = sentence_based_tokens.len();
        let ngram_count = ngram_candidates.len();
        let total_sources = (direct_count > 0) as usize + (sentence_count > 0) as usize + (ngram_count > 0) as usize;
        
        // Adaptive weights: if we have good direct continuations, prioritize them more
        let direct_weight = if direct_count > 0 {
            0.6  // Higher weight when we have direct continuations
        } else {
            0.0
        };
        
        let sentence_weight = if sentence_count > 0 {
            if direct_count > 0 {
                0.25  // Lower when we have direct continuations
            } else {
                0.5   // Higher when direct continuations are missing
            }
        } else {
            0.0
        };
        
        let ngram_weight = if ngram_count > 0 {
            if direct_count > 0 || sentence_count > 0 {
                0.15  // Lower when we have sentence-based sources
            } else {
                0.5   // Higher when sentence sources are missing
            }
        } else {
            0.0
        };
        
        // Normalize weights to sum to 1.0
        let total_weight = direct_weight + sentence_weight + ngram_weight;
        let (direct_weight, sentence_weight, ngram_weight) = if total_weight > 0.0 {
            (direct_weight / total_weight, sentence_weight / total_weight, ngram_weight / total_weight)
        } else {
            (0.0, 0.0, 0.0)
        };
        
        // Apply TF-IDF relevance scoring to boost contextually relevant candidates
        // Add direct sentence continuations (highest priority, most natural)
        for (token, score) in direct_continuations {
            let tfidf_boost = self.compute_tfidf_relevance(&token, &context_words);
            let boosted_score = score as f64 * direct_weight * tfidf_boost;
            
            if stop_at_sentence_end {
                if self.is_sentence_ender(&token) {
                    *combined_candidates.entry(token.clone()).or_insert(0.0) += boosted_score;
                }
            } else {
                if !self.is_sentence_ender(&token) {
                    *combined_candidates.entry(token.clone()).or_insert(0.0) += boosted_score;
                }
            }
        }
        
        // Add sentence-based candidates (secondary source)
        for (token, score) in sentence_based_tokens {
            let tfidf_boost = self.compute_tfidf_relevance(&token, &context_words);
            let boosted_score = score as f64 * sentence_weight * tfidf_boost;
            
            if stop_at_sentence_end {
                if self.is_sentence_ender(&token) {
                    *combined_candidates.entry(token.clone()).or_insert(0.0) += boosted_score;
                }
            } else {
                if !self.is_sentence_ender(&token) {
                    *combined_candidates.entry(token.clone()).or_insert(0.0) += boosted_score;
                }
            }
        }
        
        // Add n-gram candidates (filtered by sentences for context relevance)
        for (token, count) in ngram_candidates {
            let tfidf_boost = self.compute_tfidf_relevance(&token, &context_words);
            let boosted_score = count as f64 * ngram_weight * tfidf_boost;
            
            if stop_at_sentence_end {
                if self.is_sentence_ender(&token) {
                    *combined_candidates.entry(token.clone()).or_insert(0.0) += boosted_score;
                }
            } else {
                if !self.is_sentence_ender(&token) {
                    *combined_candidates.entry(token.clone()).or_insert(0.0) += boosted_score;
                }
            }
        }
        
        // If we have combined candidates, use them
        if !combined_candidates.is_empty() {
            let combined_vec: Vec<(String, u32)> = combined_candidates
                .into_iter()
                .map(|(token, score)| (token, score as u32))
                .collect();
            return self.select_from_continuations(&combined_vec, stop_at_sentence_end);
        }
        
        // Fallback: If no sentence-filtered n-grams, try unfiltered multi-gram ensemble
        // Use weighted combination from all n-gram sizes
        let mut fallback_candidates: HashMap<String, f64> = HashMap::new();
        
        for &ngram_size in &self.ngram_sizes {
            let ngram_weight = *self.ngram_weights.get(&ngram_size).unwrap_or(&0.0);
            if ngram_weight == 0.0 {
                continue;
            }
            
            let ngram_contexts = match self.multi_ngram_contexts.get(&ngram_size) {
                Some(ctx) => ctx,
                None => continue,
            };
            
            // Strategy 1: Try full n-gram context (n-1 tokens)
            if context.len() >= ngram_size - 1 {
                let context_key: Vec<String> = context[context.len().saturating_sub(ngram_size - 1)..].to_vec();
                
                if let Some(continuations) = ngram_contexts.get(&context_key) {
                    for (token, count) in continuations {
                        let tfidf_boost = self.compute_tfidf_relevance(&token, &context_words);
                        *fallback_candidates.entry(token.clone()).or_insert(0.0) += (*count as f64) * ngram_weight * tfidf_boost;
                    }
                }
            }
            
            // Strategy 2: Backoff to (n-2) context (if n > 2)
            if fallback_candidates.is_empty() && ngram_size > 2 && context.len() >= ngram_size - 2 {
                let context_key: Vec<String> = context[context.len().saturating_sub(ngram_size - 2)..].to_vec();
                
                for (ctx_key, continuations) in ngram_contexts {
                    if ctx_key.len() >= context_key.len() {
                        let ctx_suffix = &ctx_key[ctx_key.len() - context_key.len()..];
                        if ctx_suffix == context_key.as_slice() {
                            for (token, count) in continuations {
                                let tfidf_boost = self.compute_tfidf_relevance(&token, &context_words);
                                *fallback_candidates.entry(token.clone()).or_insert(0.0) += (*count as f64) * ngram_weight * tfidf_boost;
                            }
                        }
                    }
                }
            }
            
            // Strategy 3: Backoff to bigram (last token only)
            if fallback_candidates.is_empty() && context.len() >= 1 {
                let last_token = &context[context.len() - 1];
                
                for (ctx_key, continuations) in ngram_contexts {
                    if !ctx_key.is_empty() && ctx_key[ctx_key.len() - 1] == *last_token {
                        for (token, count) in continuations {
                            let tfidf_boost = self.compute_tfidf_relevance(&token, &context_words);
                            *fallback_candidates.entry(token.clone()).or_insert(0.0) += (*count as f64) * ngram_weight * tfidf_boost;
                        }
                    }
                }
            }
        }
        
        if !fallback_candidates.is_empty() {
            let fallback_vec: Vec<(String, u32)> = fallback_candidates
                .into_iter()
                .map(|(token, score)| (token, score as u32))
                .collect();
            return self.select_from_continuations(&fallback_vec, stop_at_sentence_end);
        }

        // Strategy 4: Final backoff to unigram (most frequent words overall)
        // This uses Laplace smoothing - all words have at least a small probability
        if !self.unigram_counts.is_empty() {
            let unigram_candidates: Vec<(String, u32)> = self.unigram_counts
                .iter()
                .map(|(token, count)| (token.clone(), *count))
                .collect();
            
            return self.select_from_continuations(&unigram_candidates, stop_at_sentence_end);
        }

        None
    }

    fn select_from_continuations(&self, continuations: &[(String, u32)], stop_at_sentence_end: bool) -> Option<String> {
        // Filter out sentence enders if we're not ready to stop
        let candidates: Vec<_> = if stop_at_sentence_end {
            // Prefer sentence enders
            continuations.iter()
                .filter(|(token, _)| self.is_sentence_ender(token))
                .collect()
        } else {
            // Avoid sentence enders unless it's appropriate
            continuations.iter()
                .filter(|(token, _)| !self.is_sentence_ender(token))
                .collect()
        };

        let candidates_to_use = if candidates.is_empty() {
            continuations.iter().collect()
        } else {
            candidates
        };

        if candidates_to_use.is_empty() {
            return None;
        }

        // Apply Laplace smoothing (add-one smoothing) for better probability estimation
        let mut smoothed_candidates: Vec<(String, f64)> = candidates_to_use
            .iter()
            .map(|(token, count)| {
                // Laplace smoothing: P(w|c) = (count(w,c) + 1) / (count(c) + |V|)
                let smoothed_count = (*count as f64) + 1.0;
                (token.clone(), smoothed_count)
            })
            .collect();

        // Sort by probability (descending) for top-k sampling
        smoothed_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-k filtering: only consider top-k most likely tokens
        if self.top_k > 0 && smoothed_candidates.len() > self.top_k {
            smoothed_candidates.truncate(self.top_k);
        }

        // Apply temperature scaling to probabilities
        // Lower temperature = sharper distribution (more deterministic)
        // Higher temperature = flatter distribution (more random)
        let temperature = self.temperature.max(0.01); // Prevent division by zero
        let temperature_scaled: Vec<(String, f64)> = smoothed_candidates
            .iter()
            .map(|(token, weight)| {
                // Apply temperature: weight^(1/temperature)
                // For temperature < 1, this amplifies differences
                // For temperature > 1, this reduces differences
                let scaled_weight = weight.powf(1.0 / temperature);
                (token.clone(), scaled_weight)
            })
            .collect();

        // Calculate total weight for normalization
        let total_weight: f64 = temperature_scaled.iter().map(|(_, weight)| *weight).sum();
        
        if total_weight == 0.0 {
            return None;
        }

        // Weighted random selection based on temperature-scaled probabilities
        let mut rng = rand::thread_rng();
        let random_value = rng.gen::<f64>() * total_weight;
        
        let mut cumulative_weight = 0.0;
        for (token, weight) in &temperature_scaled {
            cumulative_weight += weight;
            if random_value <= cumulative_weight {
                return Some(token.clone());
            }
        }

        // Fallback to most frequent (shouldn't reach here, but safety)
        candidates_to_use.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(token, _)| token.clone())
    }

    fn is_question_word(&self, word: &str) -> bool {
        let question_words = vec!["who", "what", "where", "when", "why", "how", "which", "whose", "whom"];
        question_words.contains(&word.to_lowercase().as_str())
    }

    /// Set the temperature for sampling (controls randomness)
    /// - temperature < 1.0: more deterministic, focused on high-probability tokens
    /// - temperature = 1.0: normal randomness
    /// - temperature > 1.0: more random, flatter distribution
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature.max(0.01); // Prevent division by zero
    }

    /// Set the top-k value for sampling (only consider top-k most likely tokens)
    /// - top_k = 0: no limit, consider all candidates
    /// - top_k > 0: only consider top-k candidates (reduces low-quality outputs)
    pub fn set_top_k(&mut self, top_k: usize) {
        self.top_k = top_k;
    }


    /// Generate power set (all possible subsets) of words
    /// Returns all non-empty subsets of the input words
    fn generate_power_set(words: &[String]) -> Vec<Vec<String>> {
        let n = words.len();
        let mut power_set = Vec::new();
        
        // Generate all 2^n - 1 non-empty subsets (exclude empty set)
        for i in 1..(1 << n) {
            let mut subset = Vec::new();
            for j in 0..n {
                if (i >> j) & 1 == 1 {
                    subset.push(words[j].clone());
                }
            }
            power_set.push(subset);
        }
        
        power_set
    }
    
    /// Check if a subset of words appears in a sentence (words appear together)
    fn subset_in_sentence(subset: &[String], sentence_words: &[String]) -> bool {
        // Convert sentence to set for faster lookup
        let sentence_set: std::collections::HashSet<&String> = sentence_words.iter().collect();
        
        // All words in subset must be in sentence
        subset.iter().all(|word| sentence_set.contains(word))
    }
    
    /// Find similar contexts using power set matching
    /// Generates all subsets of prompt words and matches them to sentences
    /// Sentences matching more subsets get higher weight - simpler and more effective!
    fn find_similar_contexts(&self, query_words: &[String]) -> Vec<(usize, f64)> {
        let mut context_scores: HashMap<usize, f64> = HashMap::new();
        
        // Normalize query words (lowercase, extract base word)
        let normalized_query: Vec<String> = query_words.iter()
            .map(|w| self.extract_word(w).to_lowercase())
            .collect();
        
        if normalized_query.is_empty() {
            return Vec::new();
        }
        
        // Generate power set: all possible subsets of query words
        let power_set = Self::generate_power_set(&normalized_query);
        
        // Score each sentence by counting how many subsets it matches
        for (ctx_idx, context) in self.contexts.iter().enumerate() {
            // Extract all words from this sentence (normalized)
            let sentence_words: Vec<String> = context.tokens.iter()
                .map(|t| self.extract_word(t).to_lowercase())
                .collect();
            
            // Count how many subsets from power set appear in this sentence
            let mut matching_subsets = 0;
            let mut total_subset_weight = 0.0;
            
            for subset in &power_set {
                if Self::subset_in_sentence(subset, &sentence_words) {
                    matching_subsets += 1;
                    // Larger subsets get more weight (exponential)
                    // Subset of size 1 = 1 point, size 2 = 4 points, size 3 = 9 points, etc.
                    let subset_size = subset.len() as f64;
                    total_subset_weight += subset_size * subset_size;
                }
            }
            
            // Score = number of matching subsets + weighted sum of subset sizes
            // This naturally rewards sentences with more word combinations
            let score = (matching_subsets as f64) * 2.0 + total_subset_weight;
            
            if score > 0.0 {
                context_scores.insert(ctx_idx, score);
            }
        }
        
        // Convert to sorted vector (by score, descending)
        let mut scored_contexts: Vec<(usize, f64)> = context_scores.into_iter().collect();
        scored_contexts.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // #1 PREFERENCE: Power set matching - if we found good matches, use them
        if !scored_contexts.is_empty() && scored_contexts[0].1 > 1.0 {
            // Good matches found with power set - return them
            return scored_contexts.into_iter().take(30).collect();
        }
        
        // #2 PREFERENCE: Fallback to TF-IDF vector similarity
        // Use when power set matching doesn't find strong matches
        let tfidf_results = self.find_similar_contexts_tfidf(&normalized_query);
        
        if !tfidf_results.is_empty() {
            // Combine power set results (if any) with TF-IDF results
            // TF-IDF scores are typically 0-1, so scale them to be comparable
            let mut combined: Vec<(usize, f64)> = scored_contexts;
            
            for (idx, tfidf_score) in tfidf_results {
                // Scale TF-IDF score to be comparable (multiply by 10 to match power set scale)
                let scaled_score = tfidf_score * 10.0;
                
                // Add or update score (take maximum)
                if let Some(existing) = combined.iter_mut().find(|(i, _)| *i == idx) {
                    existing.1 = existing.1.max(scaled_score);
                } else {
                    combined.push((idx, scaled_score));
                }
            }
            
            // Sort and return top matches
            combined.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            
            return combined.into_iter().take(30).collect();
        }
        
        // Return power set results even if weak (better than nothing)
        scored_contexts.into_iter().take(30).collect()
    }
    
    /// Get tokens from top matching sentences for generation
    /// This uses the best matching sentences as a source for tokens
    /// IMPROVED: Prioritizes tokens that appear near query words in sentences
    fn get_tokens_from_matching_sentences(&self, query_words: &[String]) -> Vec<(String, u32)> {
        let mut token_scores: HashMap<String, u32> = HashMap::new();
        let mut token_frequency: HashMap<String, u32> = HashMap::new(); // Count how many sentences contain each token
        
        // Normalize query words for matching
        let normalized_query: Vec<String> = query_words.iter()
            .map(|w| w.to_lowercase())
            .collect();
        
        // Find top matching sentences
        let top_sentences = self.find_similar_contexts(query_words);
        
        // Extract tokens from top matching sentences with proximity-based scoring
        for (sentence_idx, match_score) in top_sentences {
            if let Some(context) = self.contexts.get(sentence_idx) {
                // Higher match score = higher base weight for tokens from this sentence
                let base_weight = (match_score * 5.0) as u32;
                
                // Find positions of query words in this sentence
                let sentence_words: Vec<String> = context.tokens.iter()
                    .map(|t| self.extract_word(t).to_lowercase())
                    .collect();
                
                let mut query_word_positions = Vec::new();
                for (pos, word) in sentence_words.iter().enumerate() {
                    if normalized_query.contains(word) {
                        query_word_positions.push(pos);
                    }
                }
                
                for (token_pos, token) in context.tokens.iter().enumerate() {
                    let word = self.extract_word(token).to_lowercase();
                    
                    // Skip question words and query words themselves
                    if self.is_question_word(&word) || normalized_query.contains(&word) {
                        continue;
                    }
                    
                    // Count frequency: how many matching sentences contain this token
                    *token_frequency.entry(token.clone()).or_insert(0) += 1;
                    
                    // Base weight from sentence match quality
                    let mut token_weight = base_weight;
                    
                    // PROXIMITY BONUS: Tokens near query words get much higher scores
                    // This ensures words like "Mia" (before "was getting ready") rank highest
                    if !query_word_positions.is_empty() {
                        let first_query_pos = query_word_positions[0];
                        let min_distance = query_word_positions.iter()
                            .map(|&qpos| (token_pos as i32 - qpos as i32).abs())
                            .min()
                            .unwrap_or(100);
                        
                        // EXTRA BONUS: Words right before the first query word (typically the answer for "who/what")
                        // This is especially important for questions like "who was getting ready" -> "Mia"
                        let mut proximity_multiplier = 1.0;
                        if token_pos + 1 == first_query_pos {
                            // Token appears right before first query word - this is likely the answer!
                            proximity_multiplier = 20.0; // Very high bonus
                        } else if min_distance <= 2 {
                            // Closer tokens get exponentially higher scores
                            // Distance 0 (right next to) = 10x bonus, distance 1 = 5x, distance 2 = 2x
                            proximity_multiplier = match min_distance {
                                0 => 10.0,  // Right next to query word
                                1 => 5.0,   // One word away
                                2 => 2.0,   // Two words away
                                _ => 1.0,
                            };
                        }
                        
                        token_weight = ((token_weight as f64) * proximity_multiplier) as u32;
                    }
                    
                    *token_scores.entry(token.clone()).or_insert(0) += token_weight;
                }
            }
        }
        
        // Apply frequency boost: tokens that appear in more matching sentences get higher scores
        // BUT: Prioritize tokens from sentences with more unique matches over frequency
        // Frequency multiplier is smaller to avoid over-weighting repeated words
        for (token, score) in token_scores.iter_mut() {
            let freq = *token_frequency.get(token).unwrap_or(&0);
            // Reduced frequency multiplier: prioritize unique matches over frequency
            // This ensures different tokens from prompt rank higher than same word repeated
            let freq_multiplier = 1.0 + (freq as f64 * 0.5); // Reduced from 2.0 to 0.5
            *score = ((*score as f64) * freq_multiplier) as u32;
        }
        
        // Convert to vector
        token_scores.into_iter().collect()
    }
    
    /// Find words that appear in similar contexts to the query
    /// Uses context windows to find related words
    /// Uses wildcards to our favor: finds similar question-answer patterns
    /// NOW USES SENTENCE MATCHING: matches all words with all sentences
    fn find_contextual_candidates(&self, wildcard_pos: usize, query: &[String]) -> Vec<(String, u32)> {
        let mut candidates: Vec<(String, u32)> = Vec::new();
        
        // Get non-wildcard words from query for context matching
        let context_words: Vec<String> = query.iter()
            .enumerate()
            .filter(|(i, _)| *i != wildcard_pos)
            .map(|(_, token)| self.extract_word(token).to_lowercase())
            .collect();
        
        // PRIMARY METHOD: Get tokens directly from top matching sentences
        // This uses sentence matching (all words matched with all sentences)
        // Matches ALL words with ALL sentences, picks sentences that match the most
        let sentence_tokens = self.get_tokens_from_matching_sentences(&context_words);
        for (token, score) in sentence_tokens {
            if let Some(existing) = candidates.iter_mut().find(|(t, _)| t == &token) {
                existing.1 += score; // Accumulate scores
            } else {
                candidates.push((token, score));
            }
        }
        
        // SECONDARY METHOD: WILDCARD ADVANTAGE - Find contexts that match the question pattern
        // Look for sentences that contain the same non-wildcard words
        // This finds similar question-answer patterns in training data
        let similar_contexts = self.find_similar_contexts(&context_words);
        
        // Extract words from similar contexts that could fill the wildcard
        // This adds additional context-aware scoring on top of sentence matching
        for (ctx_idx, score) in similar_contexts {
            if let Some(context) = self.contexts.get(ctx_idx) {
                // Look for words in this context that appear near our context words
                for (i, token) in context.tokens.iter().enumerate() {
                    let word = self.extract_word(token).to_lowercase();
                    
                    // Skip if it's a question word or already in query
                    if self.is_question_word(&word) || context_words.contains(&word) {
                        continue;
                    }
                    
                    // Check if this word appears in a relevant position
                    // (near other query words in the context)
                    let mut relevance = (score as u32).max(1); // Base relevance from sentence match
                    
                    // Check words before and after in the context
                    let window_start = i.saturating_sub(3);
                    let window_end = (i + 4).min(context.tokens.len());
                    
                    for j in window_start..window_end {
                        if j != i {
                            let nearby_word = self.extract_word(&context.tokens[j]).to_lowercase();
                            if context_words.contains(&nearby_word) {
                                relevance += 5; // Higher score if near query words (stronger signal)
                            }
                        }
                    }
                    
                    // Bonus: if this word appears early in a highly-relevant context, boost it
                    if score >= 2.0 && i < 5 {
                        relevance += 3;
                    }
                    
                    // Add candidate with relevance score (accumulate with sentence matching scores)
                    if let Some(existing) = candidates.iter_mut().find(|(w, _)| w == token) {
                        existing.1 += relevance; // Accumulate scores
                    } else {
                        candidates.push((token.clone(), relevance));
                    }
                }
            }
        }
        
        // Also use context windows: find words that appear in similar contexts
        // This leverages the "last used" context information
        for query_word in &context_words {
            if let Some(windows) = self.context_windows.get(query_word) {
                for (before, after) in windows {
                    // Look for words that appear after this query word in training data
                    // These are words that were "last used" after this query word
                    if !after.is_empty() {
                        for (idx, token) in after.iter().take(5).enumerate() {
                            let word = self.extract_word(token).to_lowercase();
                            if !self.is_question_word(&word) && !context_words.contains(&word) {
                                // Closer words get higher scores (more relevant)
                                let proximity_score = (5 - idx) as u32;
                                if let Some(existing) = candidates.iter_mut().find(|(w, _)| w == token) {
                                    existing.1 += proximity_score;
                                } else {
                                    candidates.push((token.clone(), proximity_score));
                                }
                            }
                        }
                    }
                    
                    // Also check words before - these might be part of a phrase
                    if !before.is_empty() && before.len() >= 2 {
                        // Look at the last few words before
                        for token in before.iter().rev().take(2) {
                            let word = self.extract_word(token).to_lowercase();
                            if !self.is_question_word(&word) && !context_words.contains(&word) {
                                if let Some(existing) = candidates.iter_mut().find(|(w, _)| w == token) {
                                    existing.1 += 1;
                                } else {
                                    candidates.push((token.clone(), 1));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        candidates
    }

    fn find_matching_ngrams_with_context(&self, query: &[String], wildcard_pos: usize) -> Vec<(String, u32)> {
        // Use words before and after the wildcard as context to predict what should replace it
        let mut candidates = Vec::new();
        
        // Strategy 1: If query length matches n-gram length, do direct matching
        if query.len() == self.n {
            for ngram in self.ngram_counts.keys() {
                if ngram.len() != self.n {
                    continue;
                }

                // Check if this n-gram matches the query pattern (non-wildcard positions must match)
                let mut matches = true;
                for (i, query_token) in query.iter().enumerate() {
                    if i == wildcard_pos {
                        continue; // Skip wildcard position - it can be anything
                    }
                    // Compare base words (ignore punctuation differences)
                    let ngram_word = self.extract_word(&ngram[i]);
                    let query_word = self.extract_word(query_token);
                    if ngram_word != query_word {
                        matches = false;
                        break;
                    }
                }

                if matches && wildcard_pos < ngram.len() {
                    let candidate = ngram[wildcard_pos].clone();
                    let count = *self.ngram_counts.get(ngram).unwrap_or(&0);
                    
                    if let Some(existing) = candidates.iter_mut().find(|(word, _)| word == &candidate) {
                        existing.1 += count;
                    } else {
                        candidates.push((candidate, count));
                    }
                }
            }
        }

        // Strategy 2: Use sliding windows - match query segments of length n against n-grams
        // This handles cases where query length != n
        if candidates.is_empty() {
            // Try windows that include the wildcard position
            for window_start in 0..=query.len().saturating_sub(self.n) {
                let window_end = window_start + self.n;
                if window_end > query.len() {
                    break;
                }
                
                // Check if wildcard is in this window
                if wildcard_pos < window_start || wildcard_pos >= window_end {
                    continue;
                }

                let query_window = &query[window_start..window_end];
                let relative_wildcard_pos = wildcard_pos - window_start;

                for ngram in self.ngram_counts.keys() {
                    if ngram.len() != self.n {
                        continue;
                    }

                    // Check if n-gram matches query window pattern
                    let mut matches = true;
                    for (i, query_token) in query_window.iter().enumerate() {
                        if i == relative_wildcard_pos {
                            continue; // Skip wildcard position
                        }
                        // Compare base words (ignore punctuation differences)
                        let ngram_word = self.extract_word(&ngram[i]);
                        let query_word = self.extract_word(query_token);
                        if ngram_word != query_word {
                            matches = false;
                            break;
                        }
                    }

                    if matches && relative_wildcard_pos < ngram.len() {
                        let candidate = ngram[relative_wildcard_pos].clone();
                        let count = *self.ngram_counts.get(ngram).unwrap_or(&0);
                        
                        if let Some(existing) = candidates.iter_mut().find(|(word, _)| word == &candidate) {
                            existing.1 += count;
                        } else {
                            candidates.push((candidate, count));
                        }
                    }
                }
            }
        }

        // Strategy 3: Use context before wildcard (fallback)
        if candidates.is_empty() && wildcard_pos > 0 {
            let context_start = wildcard_pos.saturating_sub(self.n - 1);
            let context: Vec<String> = query[context_start..wildcard_pos].to_vec();
            
            if !context.is_empty() {
                // Look for n-grams that start with this context
                for ngram in self.ngram_counts.keys() {
                    if ngram.len() < context.len() + 1 {
                        continue;
                    }

                    // Check if n-gram starts with our context
                    let context_len = context.len().min(self.n - 1);
                    if context_len > 0 && ngram.len() >= context_len + 1 {
                        let ngram_prefix = &ngram[..context_len];
                        let query_prefix = &context[context.len().saturating_sub(context_len)..];
                        
                        // Compare using base words
                        let mut matches = true;
                        for (ngram_token, query_token) in ngram_prefix.iter().zip(query_prefix.iter()) {
                            if self.extract_word(ngram_token) != self.extract_word(query_token) {
                                matches = false;
                                break;
                            }
                        }
                        
                        if matches {
                            // The word right after the context is our candidate
                            let candidate = ngram[context_len].clone();
                            let count = *self.ngram_counts.get(ngram).unwrap_or(&0);
                            
                            if let Some(existing) = candidates.iter_mut().find(|(word, _)| word == &candidate) {
                                existing.1 += count;
                            } else {
                                candidates.push((candidate, count));
                            }
                        }
                    }
                }
            }
        }

        candidates
    }

    fn format_tokens(&self, tokens: &[String]) -> String {
        if tokens.is_empty() {
            return String::new();
        }

        let mut result = String::new();
        let mut prev_token = "";
        
        for token in tokens {
            // Determine if we need a space before this token
            let needs_space = if prev_token.is_empty() {
                false // First token
            } else {
                // Don't add space before punctuation that attaches to previous word
                // But do add space for standalone punctuation or after punctuation
                let first_char = token.chars().next().unwrap_or(' ');
                let is_punctuation = !first_char.is_alphanumeric() && first_char != '\'';
                let prev_ends_punctuation = prev_token.chars().last()
                    .map(|c| !c.is_alphanumeric() && c != '\'')
                    .unwrap_or(false);
                
                // Add space unless:
                // 1. Current token starts with punctuation (it attaches to previous word)
                // 2. Previous token already ended with punctuation (already spaced)
                !is_punctuation && !prev_ends_punctuation
            };
            
            if needs_space {
                result.push(' ');
            }
            
            result.push_str(token);
            prev_token = token;
        }
        
        result
    }

    /// Direct pattern matching: Replace wildcards with * and search raw training text
    /// Uses simple linear search on sentences split by .!?
    /// Returns map of wildcard positions to answer words if pattern found, None otherwise
    fn try_direct_pattern_match(&self, query: &str, wildcard_positions: &[usize]) -> Option<HashMap<usize, String>> {
        if wildcard_positions.is_empty() {
            return None;
        }
        
        // Split query into words (simple whitespace split, lowercase)
        let query_words: Vec<&str> = query.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\''))
            .filter(|w| !w.is_empty())
            .collect();
        
        if query_words.is_empty() {
            return None;
        }
        
        // Split training text into sentences - keep track of sentence endings
        // We need to preserve the full sentence with punctuation to extract phrases until sentence end
        let mut sentences_with_endings: Vec<(String, usize)> = Vec::new();
        let mut start = 0;
        let text = &self.raw_training_text;
        
        for (i, ch) in text.char_indices() {
            if ch == '.' || ch == '!' || ch == '?' {
                // Check for ellipsis (...)
                let is_ellipsis = if ch == '.' && i + 2 < text.len() {
                    let next_chars: String = text.chars().skip(i + 1).take(2).collect();
                    next_chars == ".."
                } else {
                    false
                };
                
                let sentence = text[start..=i].trim();
                if !sentence.is_empty() {
                    sentences_with_endings.push((sentence.to_string(), i));
                }
                
                if is_ellipsis {
                    start = i + 3; // Skip the ellipsis
                } else {
                    start = i + 1;
                }
            }
        }
        
        // Also add the last sentence if text doesn't end with punctuation
        if start < text.len() {
            let sentence = text[start..].trim();
            if !sentence.is_empty() {
                sentences_with_endings.push((sentence.to_string(), text.len()));
            }
        }
        
        let mut best_matches: Vec<(HashMap<usize, String>, usize)> = Vec::new();
        
        // Search each sentence
        for (sentence, _) in &sentences_with_endings {
            // Split sentence into words (same way as query) - store as lowercase strings
            let sentence_words: Vec<String> = sentence.split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\''))
                .filter(|w| !w.is_empty())
                .map(|w| w.to_lowercase())
                .collect();
            
            if sentence_words.is_empty() {
                continue;
            }
            
            // Get original sentence words with positions for phrase extraction
            let original_words: Vec<(usize, &str)> = sentence.split_whitespace()
                .map(|w| w.trim())
                .filter(|w| !w.is_empty())
                .enumerate()
                .collect();
            
            // Try to match query pattern at every position in the sentence
            for start_idx in 0..sentence_words.len() {
                // Check if we have enough words to match the query
                if start_idx + query_words.len() > sentence_words.len() {
                    continue;
                }
                
                let mut matches = true;
                let mut match_score = 0;
                let mut answer_map: HashMap<usize, String> = HashMap::new();
                
                // Match word by word
                for (i, query_word) in query_words.iter().enumerate() {
                    let query_word_lower = query_word.to_lowercase();
                    let sentence_word = &sentence_words[start_idx + i];
                    
                    if wildcard_positions.contains(&i) {
                        // This is a wildcard position - capture everything until sentence end
                        let wildcard_word_idx = start_idx + i;
                        
                        if wildcard_word_idx < original_words.len() {
                            // Join all words from this position until the end of the sentence
                            let answer_words: Vec<&str> = original_words[wildcard_word_idx..]
                                .iter()
                                .map(|(_, word)| *word)
                                .collect();
                            
                            // Join with spaces and remove sentence ending punctuation
                            let mut answer_phrase = answer_words.join(" ");
                            
                            // Remove sentence ending punctuation (., !, ?, or ...)
                            answer_phrase = answer_phrase
                                .trim_end_matches("...")
                                .trim_end_matches(|c: char| c == '.' || c == '!' || c == '?')
                                .trim()
                                .to_string();
                            
                            answer_map.insert(i, answer_phrase);
                            match_score += 1;
                        }
                    } else {
                        // This must match exactly (case-insensitive)
                        if query_word_lower == *sentence_word {
                            match_score += 2; // Exact match gets higher score
                        } else {
                            matches = false;
                            break;
                        }
                    }
                }
                
                if matches && !answer_map.is_empty() {
                    best_matches.push((answer_map, match_score));
                }
            }
        }
        
        // Return the best match (highest score)
        if !best_matches.is_empty() {
            best_matches.sort_by(|a, b| b.1.cmp(&a.1));
            return Some(best_matches[0].0.clone());
        }
        
        None
    }

    pub fn generate_response(&self, query: &str) -> String {
        let query_tokens = self.tokenize(query);
        
        if query_tokens.is_empty() {
            return "I don't understand.".to_string();
        }

        // Find wildcard positions (question words) - calculate from words for direct pattern matching
        // Split query into words to find wildcard positions
        let query_words: Vec<&str> = query.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\''))
            .filter(|w| !w.is_empty())
            .collect();
        
        let wildcard_positions_words: Vec<usize> = query_words.iter()
            .enumerate()
            .filter(|(_, word)| self.is_question_word(word))
            .map(|(i, _)| i)
            .collect();
        
        // Also find wildcard positions in tokens (for later use)
        let wildcard_positions_tokens: Vec<usize> = query_tokens.iter()
            .enumerate()
            .filter(|(_, token)| {
                let word = self.extract_word(token);
                self.is_question_word(word)
            })
            .map(|(i, _)| i)
            .collect();

        // FIRST LAYER: Try direct pattern matching in training text (using raw query and word positions)
        // Replace wildcards with * and search for exact matches
        if !wildcard_positions_words.is_empty() {
            if let Some(answer_map) = self.try_direct_pattern_match(query, &wildcard_positions_words) {
                // Found a direct match! Replace wildcards with answer phrases from the map
                // The answer phrases go until sentence end, so we don't need continuation
                // Build response by replacing wildcard tokens with answer phrase tokens
                let mut response_tokens = Vec::new();
                let mut token_idx = 0;
                
                // Create a set of wildcard token positions for quick lookup
                let wildcard_token_set: std::collections::HashSet<usize> = wildcard_positions_tokens.iter().cloned().collect();
                
                // Map word positions to token positions
                let mut word_to_token_map: HashMap<usize, usize> = HashMap::new();
                let mut word_idx = 0;
                for (i, token) in query_tokens.iter().enumerate() {
                    let word = self.extract_word(token);
                    if !word.is_empty() {
                        word_to_token_map.insert(word_idx, i);
                        word_idx += 1;
                    }
                }
                
                while token_idx < query_tokens.len() {
                    if wildcard_token_set.contains(&token_idx) {
                        // This is a wildcard token - find which word position it corresponds to
                        // and replace with the answer phrase
                        let mut found = false;
                        for (_word_idx, &wildcard_word_pos) in wildcard_positions_words.iter().enumerate() {
                            if let Some(&token_pos) = word_to_token_map.get(&wildcard_word_pos) {
                                if token_pos == token_idx {
                                    if let Some(answer_phrase) = answer_map.get(&wildcard_word_pos) {
                                        // Tokenize the answer phrase and add all tokens
                                        let answer_tokens = self.tokenize(answer_phrase);
                                        response_tokens.extend(answer_tokens);
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if !found {
                            // Fallback: just skip the wildcard token
                            token_idx += 1;
                        } else {
                            token_idx += 1;
                        }
                    } else {
                        // Regular token - keep it
                        response_tokens.push(query_tokens[token_idx].clone());
                        token_idx += 1;
                    }
                }
                
                // The answer phrase already goes to sentence end, so just return it
                return self.format_tokens(&response_tokens);
            }
        }
        
        // Use token-based wildcard positions for the rest of the code
        let wildcard_positions = wildcard_positions_tokens;

        // If no wildcards, try simple continuation
        if wildcard_positions.is_empty() {
            let context: Vec<String> = query_tokens[..query_tokens.len().min(self.n - 1)].to_vec();
            if let Some(continuation) = self.generate_continuation(&context, false) {
                let mut response_tokens = query_tokens.clone();
                response_tokens.push(continuation);
                return self.format_tokens(&response_tokens);
            }
            return "I don't have enough information to answer that.".to_string();
        }

        // Process wildcards: predict single words to replace them
        let mut response_tokens = query_tokens.clone();
        
        // Process each wildcard position
        for &wildcard_pos in &wildcard_positions {
            // Strategy 1: Use FULL TEXT SCAN - find contextual candidates from similar contexts
            // Uses sentence matching: matches ALL words with ALL sentences, picks best matches
            let contextual_candidates = self.find_contextual_candidates(wildcard_pos, &response_tokens);
            
            // Strategy 2: Try to find matching n-grams
            let ngram_candidates = self.find_matching_ngrams_with_context(&response_tokens, wildcard_pos);
            
            // Combine candidates: contextual (from full text scan) + n-gram based
            let mut candidates = contextual_candidates;
            for (token, count) in ngram_candidates {
                if let Some(existing) = candidates.iter_mut().find(|(t, _)| t == &token) {
                    existing.1 += count; // Add n-gram count to contextual score
                } else {
                    candidates.push((token, count));
                }
            }
            
            if !candidates.is_empty() {
                // Get context words for TF-IDF relevance scoring (non-wildcard words)
                let context_words_for_tfidf: Vec<String> = response_tokens.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != wildcard_pos)
                    .map(|(_, token)| self.extract_word(token).to_lowercase())
                    .collect();
                
                // Use weighted random selection with temperature and top-k
                // Apply TF-IDF relevance boost to candidates
                let mut smoothed_candidates: Vec<(String, f64)> = candidates
                    .iter()
                    .map(|(token, count)| {
                        let smoothed_count = (*count as f64) + 1.0;
                        // Apply TF-IDF relevance boost
                        let tfidf_boost = self.compute_tfidf_relevance(token, &context_words_for_tfidf);
                        let boosted_count = smoothed_count * tfidf_boost;
                        (token.clone(), boosted_count)
                    })
                    .collect();

                smoothed_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                if self.top_k > 0 && smoothed_candidates.len() > self.top_k {
                    smoothed_candidates.truncate(self.top_k);
                }

                let temperature = self.temperature.max(0.01);
                let temperature_scaled: Vec<(String, f64)> = smoothed_candidates
                    .iter()
                    .map(|(token, weight)| {
                        let scaled_weight = weight.powf(1.0 / temperature);
                        (token.clone(), scaled_weight)
                    })
                    .collect();

                let total_weight: f64 = temperature_scaled.iter().map(|(_, weight)| *weight).sum();
                
                if total_weight > 0.0 {
                    // Weighted random selection with temperature
                    let mut rng = rand::thread_rng();
                    let random_value = rng.gen::<f64>() * total_weight;
                    
                    let mut cumulative_weight = 0.0;
                    let mut predicted_word: Option<String> = None;
                    for (token, weight) in &temperature_scaled {
                        cumulative_weight += weight;
                        if random_value <= cumulative_weight {
                            predicted_word = Some(token.clone());
                            break;
                        }
                    }
                    
                    // Fallback to most frequent if something went wrong
                    let predicted_word = predicted_word.unwrap_or_else(|| {
                        candidates.iter()
                            .max_by_key(|(_, count)| *count)
                            .map(|(word, _)| word.clone())
                            .unwrap_or_default()
                    });
                    
                    // Replace the wildcard with the predicted word (single word only)
                    if wildcard_pos < response_tokens.len() {
                        response_tokens[wildcard_pos] = predicted_word;
                    }
                }
            }
        }

        // Simple continuation generation: generate one token at a time until sentence end
        // Inspired by the reference implementation - cleaner and more straightforward
        
        // Start with context from the last part of the response (n-1 tokens for n-gram prediction)
        let mut context = if response_tokens.len() >= self.n - 1 {
            response_tokens[response_tokens.len().saturating_sub(self.n - 1)..].to_vec()
        } else {
            response_tokens.clone()
        };
        
        let max_tokens = 15; // Reduced limit for shorter, more concise responses
        
        // Generate tokens one by one, similar to reference implementation
        loop {
            // Get next token from context
            if let Some(next_token) = self.generate_continuation(&context, false) {
                let base_word = self.extract_word(&next_token);
                
                // Early stopping: stop immediately at sentence end (no minimum length requirement)
                if self.is_sentence_ender(&next_token) {
                    response_tokens.push(next_token);
                    break;
                }
                
                // Enhanced repetition detection: check for multiple patterns
                let mut should_stop = false;
                
                // Check 1: Trigram repetition (3 consecutive words)
                if response_tokens.len() >= 2 {
                    let trigram = vec![
                        self.extract_word(&response_tokens[response_tokens.len() - 2]),
                        self.extract_word(&response_tokens[response_tokens.len() - 1]),
                        base_word
                    ];
                    
                    // Check if this trigram already exists anywhere in the response
                    for i in 0..=response_tokens.len().saturating_sub(3) {
                        if i + 2 < response_tokens.len() {
                            let existing_trigram = vec![
                                self.extract_word(&response_tokens[i]),
                                self.extract_word(&response_tokens[i + 1]),
                                self.extract_word(&response_tokens[i + 2])
                            ];
                            
                            if trigram == existing_trigram {
                                should_stop = true;
                                break;
                            }
                        }
                    }
                }
                
                // Check 2: Bigram repetition (same 2-word pattern appearing 3+ times)
                if !should_stop && response_tokens.len() >= 1 {
                    let bigram = vec![
                        self.extract_word(&response_tokens[response_tokens.len() - 1]),
                        base_word
                    ];
                    
                    let mut bigram_count = 0;
                    for i in 0..response_tokens.len().saturating_sub(1) {
                        if i + 1 < response_tokens.len() {
                            let existing_bigram = vec![
                                self.extract_word(&response_tokens[i]),
                                self.extract_word(&response_tokens[i + 1])
                            ];
                            
                            if bigram == existing_bigram {
                                bigram_count += 1;
                                if bigram_count >= 2 { // Already appeared twice, this would be third
                                    should_stop = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                
                // Check 3: Same word appearing too frequently in recent tokens
                if !should_stop {
                    let recent_window = 10.min(response_tokens.len());
                    let recent_tokens: Vec<&str> = response_tokens[response_tokens.len().saturating_sub(recent_window)..]
                        .iter()
                        .map(|t| self.extract_word(t))
                        .collect();
                    
                    let word_count = recent_tokens.iter()
                        .filter(|&&w| w == base_word)
                        .count();
                    
                    // If same word appears more than 30% of recent tokens, it's repetitive
                    if word_count as f64 / recent_window as f64 > 0.3 {
                        should_stop = true;
                    }
                }
                
                if should_stop {
                    break; // Repetition detected - stop generation
                }
                
                // Add the generated token to the response
                response_tokens.push(next_token.clone());
                
                // Stop if token ends with sentence-ending punctuation (no minimum length)
                if self.is_sentence_ender(&next_token) {
                    break;
                }
                
                // Maximum length limit (no minimum requirement)
                if response_tokens.len() >= 15 {
                    break;
                }
                
                // Update context for next prediction (keep last n-1 tokens)
                context.push(next_token);
                if context.len() > self.n - 1 {
                    context = context[context.len().saturating_sub(self.n - 1)..].to_vec();
                }
                
                // Safety limit
                if response_tokens.len() > max_tokens {
                    break;
                }
            } else {
                break; // No more continuations available
            }
        }

        self.format_tokens(&response_tokens)
    }
}
