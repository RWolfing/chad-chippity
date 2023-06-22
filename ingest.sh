# Bash script to ingest data
# This involves scraping the data from the web and then cleaning up and putting in Weaviate.
# Error if any command fails
set -e
wget -r -A.html -P rtdocs https://example-jupyter-book.readthedocs.io/intro.html
python3 ingest.py