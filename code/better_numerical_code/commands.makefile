all: Ex2.1 Ex2.2 # List here all examples you completed


rust:
	maturin develop -m code/better_numerical_code/Cargo.toml

Ex2.1: rust
	@echo "Exercise 2.1"
	python3 code/better_numerical_code/triangler.py analytical_result

Ex2.2: rust
	@echo "Exercise 2.2"
	python3 code/better_numerical_code/triangler.py inspect --point 0.1 0.2 0.3

Ex2.8: rust
	@echo "Exercise 2.8"
	python3 code/better_numerical_code/triangler.py -param spherical integrate -n 10 -ppi 10000 -it naive -nc 1 -s 1337

Ex2.10: rust
	@echo "Exercise 2.10"
	python3 code/better_numerical_code/triangler.py -param spherical integrate --integrator vegas