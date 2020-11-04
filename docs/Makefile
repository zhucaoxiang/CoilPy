# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PYTHONDIR     = ../coilpy

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

docs:
	rm -rf ./api ./_build
	sphinx-apidoc -Mef -o api $(PYTHONDIR)
	for f in ./api/*.rst; do\
	    perl -pi -e 's/(module|package)$$// if $$. == 1' $$f ;\
	done
	$(MAKE) html
	cp -r examples $(BUILDDIR)/html

gh-pages: docs
	rm -rf /tmp/gh-pages	
	cp -r $(BUILDDIR)/html /tmp/gh-pages
	git checkout gh-pages
	cd .. && rm -rf * && cp -r /tmp/gh-pages/* ./ && cp /tmp/gh-pages/.nojekyll ./ \
	&& rm -rf /tmp/gh-pages && git add . && git commit -m "Updated gh-pages" && git push && git checkout master

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
