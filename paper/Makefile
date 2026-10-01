MAIN = main

.PHONY: all clean purge view

all:
	latexmk -pdf -interaction=nonstopmode -synctex=1 $(MAIN).tex

clean:
	latexmk -c

purge:
	latexmk -C
