#### crisp

(Simple (Lisp) Interpreter (in Rust))

Barely in a working state at the moment, can evaluate simple arithmatic expressions and some basic general operations. Lambda, defun, and define, are not implemented yet.

##### goals
There aren't very specific goals at the moment, however, here are some general things I'd like to be able to write in this "language"
* webserver
* advent of code solutions
* shell commands

##### repl
```
cargo run repl
```
example:
```
crisp <-
(take (repeat (* pi 2.3) 5) 2)
crisp -> (7.225663103256523,7.225663103256523)
```

##### showcase
shows some simple expression evaluation
```
cargo run showcase
```
```
Evaluating  : (/ (* (+ (abs (- 5 10)) 2) 2) 2)
Result      : 7
Evaluating  : (= (! false) (> 4 3))
Result      : true
Evaluating  : (>= 4 6)
Result      : false
Evaluating  : (* pi 2.0)
Result      : 6.283185307179586
Evaluating  : (* pi 2.6)
Result      : 8.168140899333462
Evaluating  : (time (* pi pi))
Result      : (23630,9.869604401089358)
Evaluating  : (skip (take (repeat (* 2.1 2.4) 10 ) 4) 2)
Result      : (5.04,5.04)
```
#### run files
execute crisp instructions from files
```
cargo run sample/arithmatic_test.crisp
```
