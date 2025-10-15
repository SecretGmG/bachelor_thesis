all: analytical,inspect,vegas # List here all examples you completed

dev:
	maturin develop -m code/triangler/Cargo.toml

empty:
	python3 code/triangler/triangler_cli.py

analytical:
	python3 code/triangler/triangler_cli.py analytical_result

inspect:
	python3 code/triangler/triangler_cli.py inspect --point 0.1 0.2 0.3

vegas:
	python3 code/triangler/triangler_cli.py -param spherical integrate

vegas_multi:
	python3 code/triangler/triangler_cli.py -param spherical integrate