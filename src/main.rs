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
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn generate_continuation(&self, context: &[String]) -> Option<String> {
        if context.len() < self.n - 1 {
            return None;
        }

        let context_key: Vec<String> = context[context.len().saturating_sub(self.n - 1)..].to_vec();
        
        if let Some(continuations) = self.ngram_contexts.get(&context_key) {
            if continuations.is_empty() {
                return None;
            }

            // Weighted random selection based on frequency
            let total: u32 = continuations.iter().map(|(_, count)| count).sum();
            let mut rng = (total as f64 * 0.5) as u32; // Simple deterministic selection
            for (token, count) in continuations {
                if rng < *count {
                    return Some(token.clone());
                }
                rng -= count;
            }
            // Fallback to most frequent
            continuations.iter()
                .max_by_key(|(_, count)| count)
                .map(|(token, _)| token.clone())
        } else {
            None
        }
    }

    fn find_matching_ngrams(&self, query: &[String]) -> Vec<NGram> {
        let mut matches = Vec::new();
        let question_words = vec!["who", "what", "where", "when", "why", "how", "which", "whose", "whom"];

        for ngram in self.ngram_counts.keys() {
            if ngram.len() != query.len() {
                continue;
            }

            let mut matches_query = true;
            for (i, query_token) in query.iter().enumerate() {
                let query_lower = query_token.to_lowercase();
                
                // If query token is a question word, it matches anything (wildcard)
                if question_words.contains(&query_lower.as_str()) {
                    continue;
                }
                
                // Otherwise, check if it matches the ngram token
                if ngram[i].to_lowercase() != query_lower {
                    matches_query = false;
                    break;
                }
            }

            if matches_query {
                matches.push(ngram.clone());
            }
        }

        matches
    }

    fn generate_response(&self, query: &str) -> String {
        let query_tokens = self.tokenize(query);
        
        if query_tokens.is_empty() {
            return "I don't understand.".to_string();
        }

        // Find matching n-grams
        let matches = self.find_matching_ngrams(&query_tokens);
        
        if matches.is_empty() {
            // Try to generate based on context
            let context: Vec<String> = query_tokens[..query_tokens.len().min(self.n - 1)].to_vec();
            if let Some(continuation) = self.generate_continuation(&context) {
                let mut response = query_tokens.join(" ");
                response.push(' ');
                response.push_str(&continuation);
                return response;
            }
            return "I don't have enough information to answer that.".to_string();
        }

        // Select the most frequent matching n-gram
        let best_match = matches.iter()
            .max_by_key(|ngram| {
                let ngram_ref: &Vec<String> = ngram;
                self.ngram_counts.get(ngram_ref).unwrap_or(&0)
            })
            .unwrap();

        // Generate continuation from the match
        let mut response = best_match.join(" ");
        
        // Try to generate a few more tokens
        let mut context = best_match.clone();
        for _ in 0..3 {
            if let Some(next_token) = self.generate_continuation(&context) {
                response.push(' ');
                response.push_str(&next_token);
                context.push(next_token);
                // Keep context size manageable
                if context.len() > self.n {
                    context.remove(0);
                }
            } else {
                break;
            }
        }

        response
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
