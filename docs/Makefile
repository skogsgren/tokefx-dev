# /docs/Makefile
SHELL := /bin/bash
OUT := build

.PHONY: all thesis proposal methodology clean article

all: thesis proposal methodology article

thesis:
	@set -euo pipefail; \
	tmp="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp"' EXIT; \
	mkdir -p "$(OUT)"; \
	cp references.bib "$$tmp/"; \
	cp acl_natbib.bst "$$tmp/"; \
	cp -a thesis/. "$$tmp/"; \
	(cd "$$tmp" && tectonic main.tex); \
	cp "$$tmp/main.pdf" "$(OUT)/thesis.pdf"

proposal:
	@set -euo pipefail; \
	tmp="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp"' EXIT; \
	mkdir -p "$(OUT)"; \
	cp references.bib "$$tmp/"; \
	cp -a proposal/. "$$tmp/"; \
	(cd "$$tmp" && tectonic main.tex); \
	(cd "$$tmp" && tectonic slides.tex); \
	cp "$$tmp/main.pdf" "$(OUT)/proposal.pdf"; \
	cp "$$tmp/slides.pdf" "$(OUT)/proposal_slides.pdf"

methodology:
	@set -euo pipefail; \
	tmp="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp"' EXIT; \
	mkdir -p "$(OUT)"; \
	cp references.bib "$$tmp/"; \
	cp -a update_methodology/. "$$tmp/"; \
	(cd "$$tmp" && tectonic slides.tex); \
	cp "$$tmp/slides.pdf" "$(OUT)/methodology.pdf"

article:
	@set -euo pipefail; \
	tmp="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp"' EXIT; \
	mkdir -p "$(OUT)"; \
	cp references.bib "$$tmp/"; \
	cp acl_natbib.bst "$$tmp/"; \
	cp acl.sty "$$tmp/"; \
	cp -a article/. "$$tmp/"; \
	(cd "$$tmp" && tectonic acl_latex.tex); \
	cp "$$tmp/acl_latex.pdf" "$(OUT)/article.pdf"

clean:
	@rm -rf "$(OUT)"
