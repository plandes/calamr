## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE =		python
PROJ_MODULES =		python/doc python/package python/deploy
PY_DOC_POST_BUILD_DEPS += cpgraphs
PY_TEST_ALL_TARGETS +=	aligncorp alignadhoc graphexampleshtml graphexampleseps
ADD_CLEAN +=		$(EXAMPLE_DIR) results
ADD_CLEAN_ALL +=	data corpus/micro/amr.txt ~/.cache/calamr ~/.calamrrc
VAPORIZE_DEPS +=	vaporizedep


## Project
#
ALIGN_DIR =		results/align
PY_CMR_ENV =		./env
EXAMPLE_DIR = 		example


## Includes
#
include ./zenbuild/main.mk


## Configure
#
# configure the application
.PHONY:			configapp
configapp:
			[ ! -f ~/.calamrrc ] && cp src/config/dot-calamrrc ~/.calamrrc
			if [ ! -d ~/.cache/calamr ] ; then \
				mkdir -p ~/.cache/calamr ; \
				cp -r corpus ~/.cache/calamr ; \
			fi

# recreate the micro corpus using adhoc source/summary sentences in a JSON file
.PHONY:			micro
micro:			clean
			$(eval outfile := download/micro.txt.bz2)
			@$(MAKE) pyharn ARG="mkadhoc --override calamr_corpus.name=adhoc"
			@mkdir -p download
			@( cat corpus/micro/amr.txt | bzip2 > $(outfile) )
			@$(call loginfo,created $(outfile))

## Alignment
#
# create adhoc corpus graphs for one example
.PHONY:			aligncorp
aligncorp:
			rm -rf $(EXAMPLE_DIR)
			@$(MAKE) pyharn ARG="aligncorp liu-example -o $(EXAMPLE_DIR) -f txt \
			    --override='calamr_corpus.name=adhoc,calamr_default.renderer=graphviz'"

# do not invoke directly--used by the align<corpus> targets
.PHONY:			_aligncorp
_aligncorp:
			@$(call loginfo,aligning $(CORP_CONF))
			@$(MAKE) pyharn ARG="aligncorp ALL \
				--override \
				'calamr_corpus.name=$(CORP_CONF),calamr_default.renderer=graphviz,calamr_default.flow_graph_result_caching=preemptive' \
				--rendlevel $(REND_LEVEL) --cached \
				-o $(ALIGN_DIR)/$(CORP_CONF)"

# align and generate graphs for the adhoc corpus
.PHONY:			alignadhoc
alignadhoc:
			@$(MAKE) $(PY_MAKE_ARGS) CORP_CONF=adhoc REND_LEVEL=5 _aligncorp

# align and generate graphs for the proxyreport corpus
.PHONY:			alignproxy
alignproxy:
			@$(MAKE) CORP_CONF=proxy-report REND_LEVEL=5 \
				_aligncorp > align-proxy.log 2>&1 &


## Example graphs
#
# create examples of html graphs
.PHONY:			graphexampleshtml
graphexampleshtml:
			rm -rf $(EXAMPLE_DIR)
			@$(MAKE) pyharn ARG="aligncorp ALL -o $(EXAMPLE_DIR) -r 2 -f txt \
			    --override='calamr_corpus.name=adhoc,calamr_default.renderer=graphviz'"

# create examples of graphs in latex friendly EPS
.PHONY:			graphexampleseps
graphexampleseps:
			rm -rf $(EXAMPLE_DIR)
			@$(MAKE) pyharn ARG="aligncorp ALL -o $(EXAMPLE_DIR) -r 2 -f txt \
			    --override='calamr_corpus.name=adhoc,calamr_graph_render_graphviz.extension=eps,calamr_default.renderer=graphviz'"

# copy the graphs and guide to GitHub pages
.PHONY:			cpgraphs
cpgraphs:
			@$(call loginfo,copy graphs)
			mkdir -p $(PY_DOC_BUILD_HTML)
			cp -r doc $(PY_DOC_BUILD_HTML)


## Test
#
# test: unit and integration
.PHONY:			testint
testint:
			./tests/inttest.sh

# create a virtual environment for the tests
$(PY_CMR_ENV):
			@if [ ! -d "$(PY_CMR_ENV)" ] ; then \
				echo "creating environment in $(PY_CMR_ENV)" ; \
				conda env create -f src/python/environment.yml \
					--prefix=$(PY_CMR_ENV) ; \
				$(PY_CMR_ENV)/bin/pip install plac zensols.pybuild ; \
			fi

# creates a test environment and then tests
.PHONY:			testworld
testworld:		hellfire $(PY_CMR_ENV)
			@$(call loginfo,unit and integration testing...)
			@PATH="$(PY_CMR_ENV)/bin:$(PATH)" make test
			@PATH="$(PY_CMR_ENV)/bin:$(PATH)" ./tests/inttest.sh

# # remove everything (careful)
.PHONY:			vaporizedep
vaporizedep:
			rm -fr corpus download
			git checkout corpus

# # remove everything and more (careful!)
.PHONY:			hellfire
hellfire:		vaporize
			rm -fr $(PY_CMR_ENV) $(EXAMPLE_DIR) scored.csv
