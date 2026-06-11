.PHONY: setup run all

setup:
	python -m venv venv
	venv\Scripts\python.exe -m pip install --upgrade pip
	venv\Scripts\python.exe -m pip install -r requirements.txt

run:
	venv\Scripts\python.exe run_all.py

all: setup run