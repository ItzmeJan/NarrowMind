use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Write};
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

struct LanguageModel {
    n: usize,
    ngram_counts: NGramCount,
    ngram_contexts: NGramContext,
    vocabulary: Vec<String>,
    unigram_counts: HashMap<String, u32>, // For smoothing and backoff
    total_unigrams: u32, // Total word count for probability calculations
    temperature: f64, // Controls randomness: 1.0 = normal, <1.0 = more deterministic, >1.0 = more random
    top_k: usize, // Only consider top-k most likely tokens (0 = no limit)
    // Full text context scanning
    contexts: Vec<ContextEntry>, // All sentence-level contexts from training data
    word_to_contexts: HashMap<String, Vec<usize>>, // Maps words to context indices where they appear
    context_windows: HashMap<String, Vec<(Vec<String>, Vec<String>)>>, // Word -> (before_context, after_context) pairs
}

impl LanguageModel {
    fn new(n: usize) -> Self {
        Self {
            n,
            ngram_counts: HashMap::new(),
            ngram_contexts: HashMap::new(),
            vocabulary: Vec::new(),
            unigram_counts: HashMap::new(),
            total_unigrams: 0,
            temperature: 1.0, // Default: normal randomness
            top_k: 40, // Default: consider top 40 candidates
            contexts: Vec::new(),
            word_to_contexts: HashMap::new(),
            context_windows: HashMap::new(),
        }
    }

    fn train(&mut self, text: &str) {
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
        
        if tokens.len() < self.n {
            return;
        }

        for i in 0..=tokens.len().saturating_sub(self.n) {
            let ngram: NGram = tokens[i..i + self.n].to_vec();
            
            // Count n-grams
            *self.ngram_counts.entry(ngram.clone()).or_insert(0) += 1;

            // Build context for generation (context is n-1 tokens, next token is the continuation)
            if i + self.n < tokens.len() {
                let context: Vec<String> = ngram[..self.n - 1].to_vec();
                let next_token = tokens[i + self.n].clone();
                
                let continuations = self.ngram_contexts
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

        // Build vocabulary and unigram counts (only non-question words)
        for token in &tokens {
            if !self.vocabulary.contains(token) {
                self.vocabulary.push(token.clone());
            }
            // Count unigrams for smoothing and backoff
            *self.unigram_counts.entry(token.clone()).or_insert(0) += 1;
            self.total_unigrams += 1;
        }
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

    fn generate_continuation(&self, context: &[String], stop_at_sentence_end: bool) -> Option<String> {
        if context.is_empty() {
            return None;
        }

        // NEW: Use sentence matching to get tokens from matching sentences
        // This increases accuracy by using the best matching sentences
        let context_words: Vec<String> = context.iter()
            .map(|t| self.extract_word(t).to_lowercase())
            .collect();
        
        let sentence_based_tokens = self.get_tokens_from_matching_sentences(&context_words);
        
        // If we have good sentence matches, use them as primary source
        if !sentence_based_tokens.is_empty() {
            // Convert to continuation format and use sentence-matched tokens
            let sentence_continuations: Vec<(String, u32)> = sentence_based_tokens
                .into_iter()
                .filter(|(token, _)| {
                    let word = self.extract_word(token);
                    if stop_at_sentence_end {
                        self.is_sentence_ender(token)
                    } else {
                        !self.is_sentence_ender(token)
                    }
                })
                .collect();
            
            if !sentence_continuations.is_empty() {
                // Use sentence-matched tokens with temperature for better control
                return self.select_from_continuations(&sentence_continuations, stop_at_sentence_end);
            }
        }

        // Markov chain with proper backoff strategy: try longer contexts first, then shorter
        // This implements a proper n-gram backoff model
        
        // Strategy 1: Try full n-gram context (n-1 tokens)
        if context.len() >= self.n - 1 {
            let context_key: Vec<String> = context[context.len().saturating_sub(self.n - 1)..].to_vec();
            
            if let Some(continuations) = self.ngram_contexts.get(&context_key) {
                if !continuations.is_empty() {
                    return self.select_from_continuations(continuations, stop_at_sentence_end);
                }
            }
        }

        // Strategy 2: Backoff to (n-2) context (if n > 2)
        if self.n > 2 && context.len() >= self.n - 2 {
            let context_key: Vec<String> = context[context.len().saturating_sub(self.n - 2)..].to_vec();
            
            // Find all contexts that end with our context
            let mut backoff_continuations: Vec<(String, u32)> = Vec::new();
            for (ctx_key, continuations) in &self.ngram_contexts {
                if ctx_key.len() >= context_key.len() {
                    let ctx_suffix = &ctx_key[ctx_key.len() - context_key.len()..];
                    if ctx_suffix == context_key.as_slice() {
                        for (token, count) in continuations {
                            if let Some(existing) = backoff_continuations.iter_mut().find(|(t, _)| t == token) {
                                existing.1 += count;
                            } else {
                                backoff_continuations.push((token.clone(), *count));
                            }
                        }
                    }
                }
            }
            
            if !backoff_continuations.is_empty() {
                return self.select_from_continuations(&backoff_continuations, stop_at_sentence_end);
            }
        }

        // Strategy 3: Backoff to bigram (last token only)
        if context.len() >= 1 {
            let last_token = &context[context.len() - 1];
            
            // Collect all continuations where the context ends with our last token
            let mut bigram_continuations: Vec<(String, u32)> = Vec::new();
            for (ctx_key, continuations) in &self.ngram_contexts {
                if !ctx_key.is_empty() && ctx_key[ctx_key.len() - 1] == *last_token {
                    for (token, count) in continuations {
                        if let Some(existing) = bigram_continuations.iter_mut().find(|(t, _)| t == token) {
                            existing.1 += count;
                        } else {
                            bigram_continuations.push((token.clone(), *count));
                        }
                    }
                }
            }
            
            if !bigram_continuations.is_empty() {
                return self.select_from_continuations(&bigram_continuations, stop_at_sentence_end);
            }
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


    /// Find similar contexts in training data by matching ALL words with ALL sentences
    /// Returns contexts sorted by match score (sentences that match the most words)
    /// IMPROVED: Also counts word frequency - words that match more often get higher scores
    fn find_similar_contexts(&self, query_words: &[String]) -> Vec<(usize, f64)> {
        let mut context_scores: HashMap<usize, (f64, usize)> = HashMap::new();
        
        // Normalize query words (lowercase, extract base word)
        let normalized_query: Vec<String> = query_words.iter()
            .map(|w| self.extract_word(w).to_lowercase())
            .collect();
        
        // Count frequency of each query word across all sentences
        let mut query_word_frequency: HashMap<String, u32> = HashMap::new();
        for query_word in &normalized_query {
            *query_word_frequency.entry(query_word.clone()).or_insert(0) += 1;
        }
        
        // Score each sentence by matching ALL words
        for (ctx_idx, context) in self.contexts.iter().enumerate() {
            // Extract all words from this sentence (normalized)
            let sentence_words: Vec<String> = context.tokens.iter()
                .map(|t| self.extract_word(t).to_lowercase())
                .collect();
            
            // Count unique query words that appear in this sentence
            // PRIORITIZE: Different/unique tokens matter more than same word appearing multiple times
            let mut unique_matches = std::collections::HashSet::new();
            let mut total_occurrences = 0;
            let total_query_words = normalized_query.len();
            
            for query_word in &normalized_query {
                // Count occurrences in sentence
                let occurrences_in_sentence = sentence_words.iter()
                    .filter(|w| w == &query_word)
                    .count();
                
                if occurrences_in_sentence > 0 {
                    unique_matches.insert(query_word.clone());
                    total_occurrences += occurrences_in_sentence;
                }
            }
            
            let match_count = unique_matches.len();
            
            // Calculate match ratio (percentage of unique query words found)
            if total_query_words > 0 {
                let match_ratio = match_count as f64 / total_query_words as f64;
                
                // PRIORITIZE UNIQUE MATCHES: Different tokens weighted much higher
                // Base score heavily favors unique matches (each unique match = 10 points)
                let unique_match_score = (match_count as f64) * 10.0;
                
                // Frequency bonus is small (each additional occurrence = 0.5 points)
                // This ensures sentences with more different words rank higher than
                // sentences with the same word repeated many times
                let frequency_bonus = (total_occurrences - match_count) as f64 * 0.5;
                
                // Match ratio bonus (higher ratio = better)
                let ratio_bonus = match_ratio * 5.0;
                
                // Final score: unique matches are most important
                let score = unique_match_score + frequency_bonus + ratio_bonus;
                
                // Also track exact match count for tie-breaking
                context_scores.insert(ctx_idx, (score, match_count));
            }
        }
        
        // Convert to sorted vector (by score, then by match count)
        let mut scored_contexts: Vec<(usize, f64)> = context_scores.into_iter()
            .map(|(idx, (score, _))| (idx, score))
            .collect();
        scored_contexts.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Return top matching sentences (more than before for better coverage)
        scored_contexts.into_iter().take(30).collect()
    }
    
    /// Get tokens from top matching sentences for generation
    /// This uses the best matching sentences as a source for tokens
    /// IMPROVED: Frequency-based ranking - words that appear in more matching sentences rank higher
    fn get_tokens_from_matching_sentences(&self, query_words: &[String]) -> Vec<(String, u32)> {
        let mut token_scores: HashMap<String, u32> = HashMap::new();
        let mut token_frequency: HashMap<String, u32> = HashMap::new(); // Count how many sentences contain each token
        
        // Find top matching sentences
        let top_sentences = self.find_similar_contexts(query_words);
        
        // Extract tokens from top matching sentences with frequency-based scoring
        for (sentence_idx, match_score) in top_sentences {
            if let Some(context) = self.contexts.get(sentence_idx) {
                // Higher match score = higher base weight for tokens from this sentence
                let base_weight = (match_score * 5.0) as u32;
                
                for token in &context.tokens {
                    let word = self.extract_word(token).to_lowercase();
                    
                    // Skip question words
                    if self.is_question_word(&word) {
                        continue;
                    }
                    
                    // Count frequency: how many matching sentences contain this token
                    *token_frequency.entry(token.clone()).or_insert(0) += 1;
                    
                    // Base weight from sentence match quality
                    *token_scores.entry(token.clone()).or_insert(0) += base_weight;
                }
            }
        }
        
        // Apply frequency boost: tokens that appear in more matching sentences get higher scores
        // BUT: Prioritize tokens from sentences with more unique matches over frequency
        // Frequency multiplier is smaller to avoid over-weighting repeated words
        for (token, score) in token_scores.iter_mut() {
            let freq = token_frequency.get(token).unwrap_or(&0);
            // Reduced frequency multiplier: prioritize unique matches over frequency
            // This ensures different tokens from prompt rank higher than same word repeated
            let freq_multiplier = 1.0 + (freq * 0.5) as f64; // Reduced from 2.0 to 0.5
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

    fn generate_response(&self, query: &str) -> String {
        let query_tokens = self.tokenize(query);
        
        if query_tokens.is_empty() {
            return "I don't understand.".to_string();
        }

        // Find wildcard positions (question words) - ONLY in the user's query, not in training data
        // Question words in training data are treated as regular tokens
        let wildcard_positions: Vec<usize> = query_tokens.iter()
            .enumerate()
            .filter(|(_, token)| {
                let word = self.extract_word(token);
                self.is_question_word(word)
            })
            .map(|(i, _)| i)
            .collect();

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
                // Use weighted random selection with temperature and top-k
                let mut smoothed_candidates: Vec<(String, f64)> = candidates
                    .iter()
                    .map(|(token, count)| {
                        let smoothed_count = (*count as f64) + 1.0;
                        (token.clone(), smoothed_count)
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

fn main() {
    println!("NarrowMind - N-gram Language Model");
    println!("====================================\n");

    // Load training data
    let training_data = match fs::read_to_string("input.txt") {
        Ok(content) => content,
        Err(_) => {
            eprintln!("Error: Could not read input.txt. Please ensure the file exists.");
            return;
        }
    };

    if training_data.trim().is_empty() {
        eprintln!("Error: input.txt is empty. Please add training data.");
        return;
    }

    // Initialize model with trigrams (n=3)
    let n = 3;
    println!("Training model with {}-grams...", n);
    
    let mut model = LanguageModel::new(n);
    model.train(&training_data);
    
    println!("Training complete! Vocabulary size: {}\n", model.vocabulary.len());
    println!("Enter questions (use question words like who, what, where, when, why, how as wildcards)");
    println!("Type 'quit' or 'exit' to stop.\n");

    // Interactive CLI
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() || input.trim().is_empty() {
            continue;
        }

        let query = input.trim();
        if query == "quit" || query == "exit" {
            println!("Goodbye!");
            break;
        }

        let response = model.generate_response(query);
        println!("{}", response);
        println!();
    }
}
