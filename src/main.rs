mod crisp;
use std::env::args;

use crate::crisp::Tokenizer;

macro_rules! pres {
    ($program:expr) => {
        println!("Evaluating  : {}", $program);
        let tokens = Tokenizer::init($program);
        println!("Result      : {}", tokens.unwrap());
    };
}

fn main() {
    let args: Vec<_> = args().into_iter().skip(1).collect();
    // for future
    // let mut dump_flag = false;
    for arg in args {
        match arg.as_str() {
            "repl" => {
                println!("crisp <- ");
                let mut previous = String::new();
                loop {
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input).unwrap();
                    if input.trim().is_empty() {
                        break;
                    }
                    match input.trim() {
                        "q" => {
                            println!("exiting");
                            std::process::exit(0);
                        }
                        "repeat" => {
                            input = previous;
                        }
                        _ => {}
                    }
                    previous = input.clone();
                    match Tokenizer::init(&input) {
                        Ok(v) => println!("crisp -> {v}"),
                        Err(e) => println!("eval failed -> {e}"),
                    };
                    println!("crisp <- ");
                }
            }
            "showcase" => {
                pres!("(/ (* (+ (abs (- 5 10)) 2) 2) 2)");
                pres!("(= (! false) (> 4 3))");
                pres!("(>= 4 6)");
                pres!("(* pi 2.0)");
                pres!("(* pi 2.6)");
                pres!("(time (* pi pi))");
                pres!("(skip (take (repeat (* 2.1 2.4) 10 ) 4) 2)");
            }
            _ => {
                let data = std::fs::read_to_string(&arg).unwrap();
                match Tokenizer::init(&data) {
                    Ok(v) => println!("{v}"),
                    _ => println!("eval failed for {arg}"),
                };
            }
        }
    }
}
