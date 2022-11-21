use std::borrow::Cow;
use std::collections::BTreeMap;
use std::f64::consts::{E, PI, SQRT_2};
use std::f64::EPSILON;
use std::fmt::Display;
use std::sync::RwLock;
use std::time::Duration;
use std::time::SystemTime;

use lazy_static::lazy_static;

#[derive(Debug)]
#[allow(dead_code)]
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

// impl Display for Vec<Expr> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let mut result = String::new();
//         for v in self {
//             result.push_str(&v.to_string());
//         }
//         write!(f, "{}", result)
//     }
// }

#[derive(Clone)]
pub(crate) enum Expr {
    Bool(bool),
    Sym(String),
    Int(i64),
    Float(f64),
    List(Vec<Expr>),
    Fn(fn(&[Expr]) -> Result<Expr, Err>),
    Nop,
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
        write!(
            f,
            "{}",
            match self {
                Expr::Sym(sym) => {
                    Cow::from(format!(
                        "{}",
                        CONSTANTS
                            .read()
                            .unwrap()
                            .get(sym)
                            .unwrap_or(&Expr::Sym(sym.clone()))
                    ))
                }
                Expr::Int(n) => Cow::from(n.to_string()),
                Expr::Float(n) => Cow::from(n.to_string()),
                Expr::List(list) => {
                    let expressions: Vec<String> = list.iter().map(|x| x.to_string()).collect();
                    Cow::from(format!("({})", expressions.join(",")))
                }
                Expr::Fn(_) => "Function {}".into(),
                Expr::Bool(b) => Cow::from(b.to_string()),
                Expr::Nop => "".into(),
            }
        )
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
            Self::Nop => Ok(()),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Env(BTreeMap<String, Expr>);

lazy_static! {
    pub(crate) static ref CONSTANTS: RwLock<BTreeMap<String, Expr>> = RwLock::new(BTreeMap::from(
        [
            ("pi", Expr::Float(PI)),
            ("e", Expr::Float(E)),
            ("sqrt_2", Expr::Float(SQRT_2)),
        ]
        .map(|(k, v)| (k.to_string(), v))
    ));
    pub(crate) static ref ENV: RwLock<Env> = RwLock::new(Env::default());
}

macro_rules! replace_var {
    ($token:expr) => {
        if let Expr::Sym(v) = $token {
            if let Some(v) = CONSTANTS.read().unwrap().get(v.into()) {
                v.clone()
            } else {
                Expr::Sym(v.to_string())
            }
        } else {
            $token.clone()
        }
    };
}

macro_rules! load_var {
    ($a:expr) => {{
        let a: Vec<Expr> = $a.iter().map(|x| replace_var!(x)).collect();
        a
    }};
}

impl Default for Env {
    fn default() -> Self {
        Self(BTreeMap::from(
            [
                (
                    "+",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        let a = load_var!(a);
                        if let Ok(v) = Tokenizer::parse_int(a.as_slice()) {
                            return Ok(Expr::Int(v.iter().fold(0, |s, a| a + s)));
                        }
                        if let Ok(v) = Tokenizer::parse_float(a.as_slice()) {
                            return Ok(Expr::Float(v.iter().fold(0.0, |s, a| a + s)));
                        }
                        Err(Err::Generic("Invalid numeric literal".into()))
                    }),
                ),
                (
                    "*",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        let a = load_var!(a);
                        if let Ok(v) = Tokenizer::parse_int(a.as_slice()) {
                            return Ok(Expr::Int(v.iter().product()));
                        }
                        if let Ok(v) = Tokenizer::parse_float(a.as_slice()) {
                            return Ok(Expr::Float(v.iter().product()));
                        }
                        Err(Err::Generic("Invalid numeric literal".into()))
                    }),
                ),
                (
                    "/",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        let a = load_var!(a);
                        if let Ok(v) = Tokenizer::parse_int(&a) {
                            let dividend = *v.first().ok_or_else(|| {
                                Err::Generic("must have at least two numbers to divide".into())
                            })?;
                            let quotient: i64 = v[1..].iter().product();
                            return Ok(Expr::Int(dividend / quotient));
                        }
                        if let Ok(v) = Tokenizer::parse_float(&a) {
                            let dividend = *v.first().ok_or_else(|| {
                                Err::Generic("must have at least two numbers to divide".into())
                            })?;
                            let quotient: f64 = v[1..].iter().product();
                            return Ok(Expr::Float(dividend / quotient));
                        }
                        Err(Err::Generic("Invalid numeric literal".into()))
                    }),
                ),
                (
                    "-",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        let a = load_var!(a);
                        if let Ok(v) = Tokenizer::parse_int(&a) {
                            let first = *v.first().ok_or_else(|| {
                                Err::Generic("must have at least two numbers to divide".into())
                            })?;
                            let rest: i64 = v[1..].iter().sum();
                            return Ok(Expr::Int(first - rest));
                        }
                        if let Ok(v) = Tokenizer::parse_float(&a) {
                            let first = *v.first().ok_or_else(|| {
                                Err::Generic("must have at least two numbers to divide".into())
                            })?;
                            let rest = v[1..].iter().fold(0.0, |s, a| s + a);
                            return Ok(Expr::Float(first - rest));
                        }
                        Err(Err::Generic("Invalid numeric literal".into()))
                    }),
                ),
                (
                    "abs",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        let a = load_var!(a);
                        if a.len() != 1 {
                            return Err(Err::Generic("Expected 1 number".into()));
                        }
                        if let Ok(v) = Tokenizer::parse_int(&a) {
                            return Ok(Expr::Int(v.first().unwrap().abs()));
                        }
                        if let Ok(v) = Tokenizer::parse_float(&a) {
                            return Ok(Expr::Float(v.first().unwrap().abs()));
                        }
                        unreachable!();
                    }),
                ),
                (
                    "!",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        if a.len() != 1 {
                            return Err(Err::Generic("Expected 1 argument for !".into()));
                        }
                        if let Expr::Bool(v) = a[0] {
                            return Ok(Expr::Bool(!v));
                        }
                        Err(Err::Generic("Invalid expression given to !".into()))
                    }),
                ),
                (
                    "==",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        let a = load_var!(a);
                        if a.len() != 2 {
                            return Err(Err::Generic("Expected 2 values".into()));
                        }
                        Ok(Expr::Bool(a[0] == a[1]))
                    }),
                ),
                (
                    ">",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        let a = load_var!(a);
                        if a.len() != 2 {
                            return Err(Err::Generic("Expected 2 numbers".into()));
                        }
                        if let Ok(v) = Tokenizer::parse_int(&a) {
                            return Ok(Expr::Bool(v[0] > v[1]));
                        }
                        if let Ok(v) = Tokenizer::parse_float(&a) {
                            return Ok(Expr::Bool(v[0] > v[1]));
                        }
                        unreachable!();
                    }),
                ),
                (
                    "<",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        let a = load_var!(a);
                        if a.len() != 2 {
                            return Err(Err::Generic("Expected 2 numbers".into()));
                        }
                        if let Ok(v) = Tokenizer::parse_int(&a) {
                            return Ok(Expr::Bool(v[0] < v[1]));
                        }
                        if let Ok(v) = Tokenizer::parse_float(&a) {
                            return Ok(Expr::Bool(v[0] < v[1]));
                        }
                        unreachable!();
                    }),
                ),
                (
                    "<=",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        let a = load_var!(a);
                        if a.len() != 2 {
                            return Err(Err::Generic("Expected 2 numbers".into()));
                        }
                        if let Ok(v) = Tokenizer::parse_int(&a) {
                            return Ok(Expr::Bool(v[0] <= v[1]));
                        }
                        if let Ok(v) = Tokenizer::parse_float(&a) {
                            return Ok(Expr::Bool(v[0] <= v[1]));
                        }
                        unreachable!();
                    }),
                ),
                (
                    ">=",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        let a = load_var!(a);
                        if a.len() != 2 {
                            return Err(Err::Generic("Expected 2 numbers".into()));
                        }
                        if let Ok(v) = Tokenizer::parse_int(&a) {
                            return Ok(Expr::Bool(v[0] >= v[1]));
                        }
                        if let Ok(v) = Tokenizer::parse_float(&a) {
                            return Ok(Expr::Bool(v[0] >= v[1]));
                        }
                        unreachable!();
                    }),
                ),
                (
                    "time",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        if a.len() != 1 {
                            return Err(Err::Generic("Expected 1 expression".into()));
                        }
                        let start = SystemTime::now();
                        let mut res = Vec::new();
                        for v in a {
                            if let Ok(v) = Tokenizer::eval(v, None) {
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
                    }),
                ),
                (
                    "repeat",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        if a.len() != 2 {
                            return Err(Err::Generic("Expected 2 values".into()));
                        }
                        let mut res = Vec::new();
                        if let Expr::Int(v) = a[1] {
                            assert!(v >= 0);
                            for _ in 0..v {
                                res.push(a[0].clone());
                            }
                        }
                        Ok(Expr::List(res))
                    }),
                ),
                (
                    "loop",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        if a.len() != 2 {
                            return Err(Err::Generic("Expected 2 values".into()));
                        }
                        if let (Expr::Int(v), Expr::Sym(expr)) = (&a[1], &a[0]) {
                            assert!(v > &0);
                            for _ in 0..(v - 1) {
                                // let _ = Tokenizer::eval(&a[0], None);
                                let tokens: Vec<String> = expr
                                    .replace('(', " ( ")
                                    .replace(')', " ) ")
                                    .split_whitespace()
                                    .filter_map(
                                        |x| if x != "~" { Some(x.to_string()) } else { None },
                                    )
                                    .collect();
                                let (parsed, _) = Tokenizer::parse(&tokens)?;
                                let _ = Tokenizer::eval(&parsed, None)?;
                            }
                        }
                        Ok(Expr::Nop)
                    }),
                ),
                (
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
                    }),
                ),
                (
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
                    }),
                ),
                (
                    "clear",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        for v in a {
                            if let Expr::Sym(v) = v {
                                ENV.write().unwrap().0.remove(v);
                            }
                        }
                        Ok(Expr::Bool(true))
                    }),
                ),
                (
                    "define",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        if a.len() != 2 {
                            return Err(Err::Generic(
                                "tried to define procedure with more than 2 elements".into(),
                            ));
                        }
                        if let (Expr::Sym(l), Expr::Sym(r)) = (&a[0], &a[1]) {
                            // ENV.lock().unwrap().0.insert(a[0], );
                        }
                        Ok(Expr::Bool(true))
                    }),
                ),
                (
                    "bind",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        if a.len() != 2 {
                            return Err(Err::Generic(
                                "tried to define procedure with more than 2 elements".into(),
                            ));
                        }
                        let (Expr::Sym(l), r) = (a[0].clone(), &a[1]) else {
                            return Ok(Expr::Bool(false));
                        };
                        CONSTANTS
                            .write()
                            .unwrap()
                            .entry(l)
                            .and_modify(|x| *x = r.clone())
                            .or_insert_with(|| r.clone());
                        Ok(Expr::Nop)
                    }),
                ),
                (
                    "print",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        print!(
                            "{}",
                            a.iter()
                                .map(|x| x.to_string())
                                .collect::<Vec<String>>()
                                .join(" ")
                        );
                        Ok(Expr::Nop)
                    }),
                ),
                (
                    "println",
                    Expr::Fn(|a: &[Expr]| -> Result<Expr, Err> {
                        println!(
                            "{}",
                            a.iter()
                                .map(|x| x.to_string())
                                .collect::<Vec<String>>()
                                .join(" ")
                        );
                        Ok(Expr::Nop)
                    }),
                ),
            ]
            .map(|(k, v)| (k.to_string(), v)),
        ))
    }
}

pub(crate) struct Tokenizer;

impl Tokenizer {
    pub(crate) fn init(expr: &str) -> Result<Vec<Expr>, Err> {
        let mut exprs = vec![];
        let mut current = vec![];
        for line in expr.lines() {
            if line.starts_with('~') && !current.is_empty() {
                exprs.push(current.join(" "));
                current.clear();
            }
            current.push(line);
        }
        if !current.is_empty() {
            exprs.push(current.join(" "));
        }
        let res: Result<Vec<Expr>, _> = exprs
            .into_iter()
            .map(|expr| {
                let tokens: Vec<String> = expr
                    .replace('(', " ( ")
                    .replace(')', " ) ")
                    .split_whitespace()
                    .filter_map(|x| if x != "~" { Some(x.to_string()) } else { None })
                    .collect();
                let (parsed, _) = Self::parse(&tokens).unwrap();
                let result = Self::eval(&parsed, None);
                if let Ok(v) = &result {
                    ENV.write().unwrap().0.insert("ans".to_string(), v.clone());
                }
                result
            })
            .collect();
        res
    }
    pub fn eval(exp: &Expr, env: Option<&mut Env>) -> Result<Expr, Err> {
        let env = ENV.read().unwrap();
        match exp {
            Expr::Sym(v) => Ok({
                if let Some(v) = env.0.get(v) {
                    v.clone()
                } else {
                    // CONSTANTS.lock().unwrap().get(v).unwrap_or(&Expr::Sym(v.to_string())).clone()
                    Expr::Sym(v.to_string())
                }
            }),
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => Ok(exp.clone()),
            Expr::Fn(_) => Err(Err::Generic("Invalid function".into())),
            Expr::List(l) => {
                let first = l
                    .first()
                    .ok_or_else(|| Err::Generic("Expected non empty list".into()))?;
                let args = &l[1..];
                let evaled = Self::eval(first, None)?;
                match evaled {
                    Expr::Fn(f) => {
                        let evaled_arg: Result<Vec<Expr>, Err> =
                            args.iter().map(|x| Self::eval(x, None)).collect();
                        f(&evaled_arg?)
                    }
                    _ => Ok(Expr::List(l.to_vec())),
                }
            }
            Expr::Nop => Ok(Expr::Nop),
        }
    }
    fn parse(tokens: &[String]) -> Result<(Expr, &[String]), Err> {
        let (token, rest) = tokens
            .split_first()
            .ok_or_else(|| Err::Generic("Token unreachable".into()))?;
        match &token[..] {
            "(" => {
                let mut exprs = vec![];
                let mut to_be_parsed = rest;
                let mut to_procedure = false;
                let mut proc: Vec<String> = vec![];
                let mut delete = false;
                loop {
                    let (token, rest) = to_be_parsed
                        .split_first()
                        .ok_or_else(|| Err::Generic("No closing )".into()))?;
                    match token.trim() {
                        ")" => {
                            if !to_procedure {
                                return Ok((Expr::List(exprs), rest));
                            } else {
                                proc.push(")".into());
                            }
                        }
                        "[" => to_procedure = true,
                        "]" => {
                            to_procedure = false;
                            exprs.push(Expr::Sym(proc.join(" ")));
                            delete = true;
                        }
                        _ => {
                            if to_procedure {
                                proc.push(token.clone());
                            }
                        }
                    }
                    if to_be_parsed.is_empty() {
                        break Ok((Expr::Nop, &[]));
                    }
                    if !to_procedure && !delete {
                        let (parsed, left_to_parse) = Self::parse(to_be_parsed)?;
                        exprs.push(parsed);
                        to_be_parsed = left_to_parse;
                    } else {
                        to_be_parsed = &to_be_parsed[1..];
                    }
                    delete = false;
                }
            }
            ")" => Err(Err::UnevenParen(1)),
            _ => Ok((Self::parse_atom(token), rest)),
        }
    }
    pub fn parse_atom(token: &str) -> Expr {
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
        assert_eq!(vec![Expr::Int(35)], tokenize.unwrap())
    }

    #[test]
    fn constants_test() {
        let program = "(< 8.16 (* 2.6 pi))";
        let tokenize = Tokenizer::init(program);
        assert_eq!(vec![Expr::Bool(true)], tokenize.unwrap())
    }

    #[test]
    fn comparison_test_g() {
        let program = "(> 4 3)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(vec![Expr::Bool(true)], tokenize.unwrap())
    }

    #[test]
    fn comparison_test_l() {
        let program = "(< 4 3)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(vec![Expr::Bool(false)], tokenize.unwrap())
    }

    #[test]
    fn comparison_test_le() {
        let program = "(<= 4 4)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(vec![Expr::Bool(true)], tokenize.unwrap())
    }

    #[test]
    fn comparison_test_ge() {
        let program = "(>= 4 4)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(vec![Expr::Bool(true)], tokenize.unwrap())
    }

    #[test]
    fn comparison_test_eq() {
        let program = "(== 4 4)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(vec![Expr::Bool(true)], tokenize.unwrap())
    }

    #[test]
    fn list_test() {
        let program = "(skip (take (repeat (* 2.1 2.4) 10 ) 4) 2)";
        let tokenize = Tokenizer::init(program);
        assert_eq!(
            vec![Expr::List(vec![Expr::Float(5.04), Expr::Float(5.04)])],
            tokenize.unwrap()
        )
    }

    #[test]
    fn bind_var() {
        let program = r#"
            ~(bind x 40)
            ~(bind x 20)
            ~(== x 20)
            "#;
        let tokenize = Tokenizer::init(program);
        assert_eq!(vec![Expr::Bool(true)], tokenize.unwrap())
    }
}
