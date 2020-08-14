all: docs

docs: README.md

build: hoof.py
	mkdir -p dist && rm -r dist/*
	python setup.py build sdist

README.md: README.Rmd
	jupytext --from Rmd --to ipynb --output - $^ \
		| jupyter nbconvert --stdin --to markdown --execute --output $@
	

