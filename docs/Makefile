# /docs/Makefile
SHELL := /bin/bash
OUT := build

.PHONY: all thesis proposal clean

all: thesis proposal

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

clean:
	@rm -rf "$(OUT)"
