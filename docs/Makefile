SHELL := /bin/bash
OUT := build

.PHONY: all clean thesis proposal methodology article results brief

all: thesis proposal methodology article results brief

COMMON := references.bib acl_natbib.bst acl.sty

define build
	@set -euo pipefail; \
	tmp="$$(mktemp -d)"; \
	trap 'rm -rf "$$tmp"' EXIT; \
	mkdir -p "$(OUT)"; \
	cp $(COMMON) "$$tmp/" 2>/dev/null || true; \
	cp -a "$(1)/." "$$tmp/"; \
	hook="$(strip $(3))"; \
	if [[ -n "$$hook" ]]; then $$hook "$$tmp"; fi; \
	(cd "$$tmp" && $(2)); \
	for pair in $(4); do \
		src="$${pair%%:*}"; \
		dst="$${pair##*:}"; \
		cp "$$tmp/$$src" "$(OUT)/$$dst"; \
	done
endef

thesis:
	$(call build,thesis,tectonic main.tex,./scripts/build_plots,main.pdf:thesis.pdf)

proposal:
	$(call build,proposal,tectonic main.tex && tectonic slides.tex,,main.pdf:proposal.pdf slides.pdf:proposal_slides.pdf)

methodology:
	$(call build,update_methodology,tectonic slides.tex,,slides.pdf:methodology.pdf)

article:
	$(call build,article,tectonic acl_latex.tex,,acl_latex.pdf:article.pdf)

results:
	$(call build,update_results,tectonic slides.tex,,slides.pdf:results.pdf)

brief:
	$(call build,brief_presentation,tectonic slides.tex,,slides.pdf:brief.pdf)

clean:
	@rm -rf "$(OUT)"
