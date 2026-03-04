.PHONY: setup test lint run-app clean

setup:
	pip install -r requirements.txt
	nbstripout --install

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/ app/

format:
	ruff format src/ tests/ app/

run-app:
	streamlit run app/streamlit_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f data/recommendations.db
