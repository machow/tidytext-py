all: docs

docs: README.md

build: 	tidytext.py
	mkdir -p dist && rm -rf dist/*
	python setup.py build sdist

README.md: README.Rmd
	jupytext --from Rmd --to ipynb --output - $^ | \
		jupyter nbconvert --stdin --to markdown \
		--TagRemovePreprocessor.remove_input_tags="{'hide-input'}" \
	    --execute --output $@

	

