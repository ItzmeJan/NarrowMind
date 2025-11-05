use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Write};

type NGram = Vec<String>;
type NGramCount = HashMap<NGram, u32>;
type NGramContext = HashMap<Vec<String>, Vec<(String, u32)>>;

struct LanguageModel {
    n: usize,
    ngram_counts: NGramCount,
    ngram_contexts: NGramContext,
    vocabulary: Vec<String>,
}

impl LanguageModel {
    fn new(n: usize) -> Self {
        Self {
            n,
            ngram_counts: HashMap::new(),
            ngram_contexts: HashMap::new(),
            vocabulary: Vec::new(),
        }
    }

    fn train(&mut self, text: &str) {
        // Tokenize training data - all words including question words are treated as regular tokens
        // Question words in training data (like "when" in "for school when") are NOT wildcards
        let tokens = self.tokenize(text);
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

        // Build vocabulary
        for token in &tokens {
            if !self.vocabulary.contains(token) {
                self.vocabulary.push(token.clone());
            }
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
        if context.len() < self.n - 1 {
            return None;
        }

        let context_key: Vec<String> = context[context.len().saturating_sub(self.n - 1)..].to_vec();
        
        // Try exact match first
        if let Some(continuations) = self.ngram_contexts.get(&context_key) {
            if continuations.is_empty() {
                return None;
            }

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

            // Weighted selection based on frequency
            let total: u32 = candidates_to_use.iter().map(|(_, count)| *count).sum();
            if total == 0 {
                return None;
            }
            
            // Use most frequent instead of weighted random for determinism
            candidates_to_use.iter()
                .max_by_key(|(_, count)| *count)
                .map(|(token, _)| token.clone())
        } else {
            // No exact match - return None rather than using incorrect fallback
            // This ensures we only use tokens that actually follow the exact context
            None
        }
    }

    fn is_question_word(&self, word: &str) -> bool {
        let question_words = vec!["who", "what", "where", "when", "why", "how", "which", "whose", "whom"];
        question_words.contains(&word.to_lowercase().as_str())
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
        let mut result = String::new();
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 {
                // Add space before each token (punctuation is already attached to words)
                result.push(' ');
            }
            result.push_str(token);
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

        // Process wildcards: predict words to replace them
        let mut response_tokens = query_tokens.clone();
        
        // Process each wildcard position
        for &wildcard_pos in &wildcard_positions {
            // Find candidates for this wildcard position using context
            let candidates = self.find_matching_ngrams_with_context(&response_tokens, wildcard_pos);
            
            if !candidates.is_empty() {
                // Select the most frequent candidate
                let best_candidate = candidates.iter()
                    .max_by_key(|(_, count)| count)
                    .map(|(word, _)| word.clone());
                
                if let Some(predicted_word) = best_candidate {
                    // Replace the wildcard with the predicted word
                    if wildcard_pos < response_tokens.len() {
                        response_tokens[wildcard_pos] = predicted_word;
                    }
                }
            }
        }

        // Generate continuation from the completed response
        // Build context from the last part of the response
        let mut context = response_tokens.clone();
        if context.len() > self.n - 1 {
            context = context[context.len().saturating_sub(self.n - 1)..].to_vec();
        }
        
        // Generate tokens naturally until we hit a sentence boundary
        // Stop only when we generate a token ending with sentence-ending punctuation
        let max_tokens = 30; // Limit to prevent overly long responses
        let mut generated_count = 0;
        
        while generated_count < max_tokens {
            if let Some(next_token) = self.generate_continuation(&context, false) {
                response_tokens.push(next_token.clone());
                context.push(next_token.clone());
                
                // Stop immediately if the token ends with sentence-ending punctuation
                if self.is_sentence_ender(&next_token) {
                    break;
                }
                
                // Keep context size manageable (use last n-1 tokens)
                if context.len() > self.n - 1 {
                    context = context[context.len().saturating_sub(self.n - 1)..].to_vec();
                }
                
                generated_count += 1;
            } else {
                // No more continuations available - stop naturally
                break;
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
