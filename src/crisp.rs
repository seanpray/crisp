use std::collections::HashMap;
use std::f64::consts::{E, PI, SQRT_2};
use std::f64::EPSILON;
use std::fmt::Display;
use std::time::Duration;
use std::time::SystemTime;

use lazy_static::lazy_static;

#[derive(Debug)]
pub(crate) enum Err {
    Syntax(usize, usize),
    UnevenParen(usize),
    Generic(String),
}

impl Display for Err {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Err::Syntax(l, c) => write!(f, "Syntax error on line {l}, col {c}"),
            Err::UnevenParen(n) => write!(f, "Uneven parenthesis: {n}"),
            Err::Generic(r) => write!(f, "{r}"),
        }
    }
}

#[derive(Clone)]
pub(crate) enum Expr {
    Bool(bool),
    Sym(String),
    Int(i64),
    Float(f64),
    List(Vec<Expr>),
    Fn(fn(&[Expr]) -> Result<Expr, Err>),
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Expr::Bool(l), Expr::Bool(r)) => l == r,
            (Expr::Sym(l), Expr::Sym(r)) => l == r,
            (Expr::Int(l), Expr::Int(r)) => l == r,
            (Expr::Float(l), Expr::Float(r)) => (l - r).abs() < EPSILON,
            (Expr::List(l), Expr::List(r)) => l == r,
            _ => false,
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Expr::Sym(symbol) => symbol.clone(),
            Expr::Int(n) => n.to_string(),
            Expr::Float(n) => n.to_string(),
            Expr::List(list) => {
                let expressions: Vec<String> = list.iter().map(|x| x.to_string()).collect();
                format!("({})", expressions.join(","))
            }
            Expr::Fn(_) => "Function {}".to_string(),
            Expr::Bool(b) => b.to_string(),
        };
        write!(f, "{}", str)
    }
}

impl std::fmt::Debug for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(a) => f.debug_tuple("boolean").field(a).finish(),
            Self::Sym(a) => f.debug_tuple("sym").field(a).finish(),
            Self::Int(a) => f.debug_tuple("int").field(a).finish(),
            Self::Float(a) => f.debug_tuple("float").field(a).finish(),
            Self::List(a) => f.debug_tuple("list").field(a).finish(),
            Self::Fn(_) => write!(f, "function"),
        }
    }
}

lazy_static! {
    pub(crate) static ref CONSTANTS: HashMap<&'static str, Expr> = HashMap::from([
        ("pi", Expr::Float(PI)),
        ("e", Expr::Float(E)),
        ("sqrt_2", Expr::Float(SQRT_2)),
    ]);
    // pub(crate) static ref PROCEDURES
}

pub(crate) struct Env(HashMap<String, Expr>);

macro_rules! new_env {
    ($k:expr, $v:expr) => {
        ($k.to_string(), $v)
    };
}

impl Default for Env {
    fn default() -> Self {
        Self(HashMap::from([
            new_env!(
                "+",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if let Ok(v) = Tokenizer::parse_int(a) {
                        return Ok(Expr::Int(v.iter().fold(0, |s, a| a + s)));
                    }
                    if let Ok(v) = Tokenizer::parse_float(a) {
                        return Ok(Expr::Float(v.iter().fold(0.0, |s, a| a + s)));
                    }
                    Err(Err::Generic("Invalid numeric literal".into()))
                })
            ),
            new_env!(
                "*",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if let Ok(v) = Tokenizer::parse_int(a) {
                        return Ok(Expr::Int(v.iter().product()));
                    }
                    if let Ok(v) = Tokenizer::parse_float(a) {
                        return Ok(Expr::Float(v.iter().product()));
                    }
                    Err(Err::Generic("Invalid numeric literal".into()))
                })
            ),
            new_env!(
                "/",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if let Ok(v) = Tokenizer::parse_int(a) {
                        let dividend = *v.first().ok_or_else(|| {
                            Err::Generic("must have at least two numbers to divide".into())
                        })?;
                        let quotient: i64 = v[1..].iter().product();
                        return Ok(Expr::Int(dividend / quotient));
                    }
                    if let Ok(v) = Tokenizer::parse_float(a) {
                        let dividend = *v.first().ok_or_else(|| {
                            Err::Generic("must have at least two numbers to divide".into())
                        })?;
                        let quotient: f64 = v[1..].iter().product();
                        return Ok(Expr::Float(dividend / quotient));
                    }
                    Err(Err::Generic("Invalid numeric literal".into()))
                })
            ),
            new_env!(
                "-",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if let Ok(v) = Tokenizer::parse_int(a) {
                        let first = *v.first().ok_or_else(|| {
                            Err::Generic("must have at least two numbers to divide".into())
                        })?;
                        let rest: i64 = v[1..].iter().sum();
                        return Ok(Expr::Int(first - rest));
                    }
                    if let Ok(v) = Tokenizer::parse_float(a) {
                        let first = *v.first().ok_or_else(|| {
                            Err::Generic("must have at least two numbers to divide".into())
                        })?;
                        let rest = v[1..].iter().fold(0.0, |s, a| s + a);
                        return Ok(Expr::Float(first - rest));
                    }
                    Err(Err::Generic("Invalid numeric literal".into()))
                })
            ),
            new_env!(
                "abs",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 1 {
                        return Err(Err::Generic("Expected 1 number".into()));
                    }
                    if let Ok(v) = Tokenizer::parse_int(a) {
                        return Ok(Expr::Int(v.first().unwrap().abs()));
                    }
                    if let Ok(v) = Tokenizer::parse_float(a) {
                        return Ok(Expr::Float(v.first().unwrap().abs()));
                    }
                    unreachable!();
                })
            ),
            new_env!(
                "!",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 1 {
                        return Err(Err::Generic("Expected 1 argument for !".into()));
                    }
                    if let Expr::Bool(v) = a[0] {
                        return Ok(Expr::Bool(!v));
                    }
                    Err(Err::Generic("Invalid expression given to !".into()))
                })
            ),
            new_env!(
                "=",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 2 {
                        return Err(Err::Generic("Expected 2 values".into()));
                    }
                    Ok(Expr::Bool(a[0] == a[1]))
                })
            ),
            new_env!(
                ">",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 2 {
                        return Err(Err::Generic("Expected 2 numbers".into()));
                    }
                    if let Ok(v) = Tokenizer::parse_int(a) {
                        return Ok(Expr::Bool(v[0] > v[1]));
                    }
                    if let Ok(v) = Tokenizer::parse_float(a) {
                        return Ok(Expr::Bool(v[0] > v[1]));
                    }
                    unreachable!();
                })
            ),
            new_env!(
                "<",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 2 {
                        return Err(Err::Generic("Expected 2 numbers".into()));
                    }
                    if let Ok(v) = Tokenizer::parse_int(a) {
                        return Ok(Expr::Bool(v[0] < v[1]));
                    }
                    if let Ok(v) = Tokenizer::parse_float(a) {
                        return Ok(Expr::Bool(v[0] < v[1]));
                    }
                    unreachable!();
                })
            ),
            new_env!(
                "<=",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 2 {
                        return Err(Err::Generic("Expected 2 numbers".into()));
                    }
                    if let Ok(v) = Tokenizer::parse_int(a) {
                        return Ok(Expr::Bool(v[0] <= v[1]));
                    }
                    if let Ok(v) = Tokenizer::parse_float(a) {
                        return Ok(Expr::Bool(v[0] <= v[1]));
                    }
                    unreachable!();
                })
            ),
            new_env!(
                ">=",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 2 {
                        return Err(Err::Generic("Expected 2 numbers".into()));
                    }
                    if let Ok(v) = Tokenizer::parse_int(a) {
                        return Ok(Expr::Bool(v[0] >= v[1]));
                    }
                    if let Ok(v) = Tokenizer::parse_float(a) {
                        return Ok(Expr::Bool(v[0] >= v[1]));
                    }
                    unreachable!();
                })
            ),
            new_env!(
                "time",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 1 {
                        return Err(Err::Generic("Expected 1 expression".into()));
                    }
                    let start = SystemTime::now();
                    let mut res = Vec::new();
                    for v in a {
                        if let Ok(v) = Tokenizer::eval(v, &mut Env::default()) {
                            res.push(v);
                        }
                    }
                    res.insert(
                        0,
                        Expr::Sym(format!(
                            "{}",
                            SystemTime::now()
                                .duration_since(start)
                                .unwrap_or(Duration::from_secs(0))
                                .as_nanos()
                        )),
                    );
                    Ok(Expr::List(res))
                })
            ),
            new_env!(
                "repeat",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 2 {
                        return Err(Err::Generic("Expected 2 values".into()));
                    }
                    let mut res = Vec::new();
                    if let Expr::Int(v) = a[1] {
                        for _ in 0..v {
                            if let Ok(v) = Tokenizer::eval(&a[0], &mut Env::default()) {
                                res.push(v);
                            }
                        }
                    }
                    Ok(Expr::List(res))
                })
            ),
            new_env!(
                "skip",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 2 {
                        return Err(Err::Generic("Expected 2 values".into()));
                    }
                    let mut res = Vec::new();
                    if let Expr::Int(v) = a[1] {
                        if let Expr::List(l) = &a[0] {
                            res = l[v as usize..].to_vec();
                        }
                    }
                    Ok(Expr::List(res))
                })
            ),
            new_env!(
                "take",
                Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                    if a.len() != 2 {
                        return Err(Err::Generic("Expected 2 values".into()));
                    }
                    let mut res = Vec::new();
                    if let Expr::Int(v) = a[1] {
                        if let Expr::List(l) = &a[0] {
                            res = l[..v as usize].to_vec();
                        }
                    }
                    Ok(Expr::List(res))
                })
            ),
        ]))
    }
}

pub(crate) struct Tokenizer;

impl Tokenizer {
    pub(crate) fn init(expr: &str) -> Result<Expr, Err> {
        let tokens: Vec<String> = expr
            .replace('(', " ( ")
            .replace(')', " ) ")
            .split_whitespace()
            .map(|x| x.to_string())
            .collect();
        let (parsed, _) = Self::parse(&tokens)?;
        Self::eval(&parsed, &mut Env::default())
    }
    fn eval(exp: &Expr, env: &mut Env) -> Result<Expr, Err> {
        match exp {
            Expr::Sym(v) => Ok(env
                .0
                .get(v)
                .unwrap_or(&Expr::Sym(v.to_string())).clone()),
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => Ok(exp.clone()),
            Expr::Fn(_) => Err(Err::Generic("Invalid function".into())),
            Expr::List(l) => {
                let first = l
                    .first()
                    .ok_or_else(|| Err::Generic("Expected non empty list".into()))?;
                let args = &l[1..];
                let evaled = Self::eval(first, env)?;
                match evaled {
                    Expr::Fn(f) => {
                        let evaled_arg: Result<Vec<Expr>, Err> =
                            args.iter().map(|x| Self::eval(x, env)).collect();
                        f(&evaled_arg?)
                    }
                    _ => Ok(Expr::List(l.to_vec())),
                }
            }
        }
    }
    fn parse(tokens: &[String]) -> Result<(Expr, &[String]), Err> {
        let (token, rest) = tokens
            .split_first()
            .ok_or_else(|| Err::Generic("Token unreachable".into()))?;
        match &token[..] {
            "(" => {
                let mut exprs = Vec::new();
                let mut to_be_parsed = rest;
                loop {
                    let (token, rest) = to_be_parsed
                        .split_first()
                        .ok_or_else(|| Err::Generic("No closing )".into()))?;
                    if token.trim() == ")" {
                        return Ok((Expr::List(exprs), rest));
                    }
                    let (parsed, left_to_parse) = Self::parse(to_be_parsed)?;
                    exprs.push(parsed);
                    to_be_parsed = left_to_parse;
                }
            }
            ")" => Err(Err::UnevenParen(1)),
            _ => Ok((Self::parse_atom(token), rest)),
        }
    }
    fn parse_atom(token: &str) -> Expr {
        match token {
            "true" => Expr::Bool(true),
            "false" => Expr::Bool(false),
            _ => {
                let num: Result<i64, _> = token.parse();
                if let Ok(v) = num {
                    return Expr::Int(v);
                }
                let num: Result<f64, _> = token.parse();
                if let Ok(v) = num {
                    return Expr::Float(v);
                }
                if let Some(v) = CONSTANTS.get(token) {
                    return v.clone();
                }
                Expr::Sym(token.to_string())
            }
        }
    }
    pub(crate) fn parse_int(args: &[Expr]) -> Result<Vec<i64>, Err> {
        args.iter()
            .map(|x| match x {
                Expr::Int(n) => Ok(*n),
                _ => Err(Err::Generic("Invalid integer literal".into())),
            })
            .collect()
    }
    pub(crate) fn parse_float(args: &[Expr]) -> Result<Vec<f64>, Err> {
        args.iter()
            .map(|x| match x {
                Expr::Float(n) => Ok(*n),
                _ => Err(Err::Generic("Invalid floating point literal".into())),
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn error_test() {
        let error = Err::Generic("L".into());
        println!("{}", error)
    }

    #[test]
    fn parse_test() {
        let tokens = Tokenizer::init("(begin (define r 10) (* pi (* r r)))");
        println!("{:?}", tokens);
    }

    #[test]
    fn eval_test() {
        let program = "(* 5 (/ (* (+ (abs (- 5 10)) 2) 2) 2))";
        let tokenize = Tokenizer::init(program);
        assert_eq!(Expr::Int(35), tokenize.unwrap())
    }

    #[test]
    fn constants_test() {
        let program = "(< 8.16 (* 2.6 pi))";
        let tokenize = Tokenizer::init(program);
        assert_eq!(Expr::Bool(true), tokenize.unwrap())
    }

    #[test]
    fn comparison_test_g() {
        let program = "(> 4 3)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(Expr::Bool(true), tokenize.unwrap())
    }

    #[test]
    fn comparison_test_l() {
        let program = "(< 4 3)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(Expr::Bool(false), tokenize.unwrap())
    }

    #[test]
    fn comparison_test_le() {
        let program = "(<= 4 4)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(Expr::Bool(true), tokenize.unwrap())
    }

    #[test]
    fn comparison_test_ge() {
        let program = "(>= 4 4)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(Expr::Bool(true), tokenize.unwrap())
    }

    #[test]
    fn comparison_test_eq() {
        let program = "(= 4 4)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(Expr::Bool(true), tokenize.unwrap())
    }

    #[test]
    fn list_test() {
        let program = "(skip (take (repeat (* 2.1 2.4) 10 ) 4) 2)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(Expr::List(vec![Expr::Float(5.04), Expr::Float(5.04)]), tokenize.unwrap())
    }
}
