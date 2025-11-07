mod language_model;

use language_model::LanguageModel;
use std::fs;
use std::io::{self, BufRead, Write};

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

    // Initialize model with multi-gram ensemble (bigrams + trigrams)
    // Default weights: trigram=0.625, bigram=0.375 (normalized from 0.5 and 0.3)
    let n = 3; // Primary n-gram size (for backward compatibility)
    println!("Training multi-gram ensemble model...");
    
    let mut model = LanguageModel::new(n);
    println!("  - N-gram sizes: {:?}", model.ngram_sizes);
    println!("  - Weights: bigram={:.2}, trigram={:.2}", 
             model.ngram_weights.get(&2).unwrap_or(&0.0),
             model.ngram_weights.get(&3).unwrap_or(&0.0));
    
    model.train(&training_data);
    
    println!("Training complete! Vocabulary size: {}", model.vocabulary.len());
    println!("  - Multi-gram contexts trained for sizes: {:?}\n", model.ngram_sizes);
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
