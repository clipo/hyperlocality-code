.PHONY: install reproduce lint docker clean

install:        ## install Python dependencies (pinned)
	pip install -r requirements-lock.txt

reproduce:      ## run the full analysis pipeline (-> output/, figures/)
	bash reproduce.sh

lint:           ## static-check the analysis code with ruff
	ruff check src/

docker:         ## build and run the pipeline in a container
	docker build -t hyperlocality-code .
	docker run --rm -v "$$PWD/output:/work/output" -v "$$PWD/figures:/work/figures" hyperlocality-code

clean:          ## remove generated outputs
	rm -rf output/*.txt figures/*.png figures/*.pdf figures/*.svg figures/_tiles src/__pycache__
