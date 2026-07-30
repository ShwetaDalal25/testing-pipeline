.PHONY: run test docker-build docker-run

run:
	streamlit run app.py

test:
	python -m py_compile app.py
	pytest -q
