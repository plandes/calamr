## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE =		python
PROJ_MODULES =		python/doc python/package python/deploy python/envdist
PY_DOC_POST_BUILD_DEPS += cpgraphs
PY_TEST_PRE_TARGETS +=	$(MICRO_CORP_FILE)
PY_TEST_ALL_TARGETS +=	aligncorp alignadhoc graphexampleshtml graphexampleseps
ADD_CLEAN +=		$(EXAMPLE_DIR)
ADD_CLEAN_ALL +=	data download corpus/micro/amr.txt ~/.cache/calamr ~/.calamrrc
VAPORIZE_DEPS +=	vaporizedep


## Project
#
EXAMPLE_DIR ?= 		align-example
MICRO_CORP_FILE ?=	download/micro.txt.bz2


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
$(MICRO_CORP_FILE):
			@mkdir -p corpus/amr-rel
			$(eval outfile := download/micro.txt.bz2)
			@$(MAKE) pyharn ARG="mkadhoc"
			@mkdir -p download
			@( cat corpus/micro/amr.txt | bzip2 > $(MICRO_CORP_FILE) )
			@$(call loginfo,created $(MICRO_CORP_FILE))
.PHONY:			micro
micro:			$(MICRO_CORP_FILE)


## Alignment
#
# create adhoc corpus graphs for one example
.PHONY:			aligncorp
aligncorp:
			@rm -rf $(EXAMPLE_DIR)
			@$(MAKE) $(PY_MAKE_ARGS) pyharn \
				ARG="align -k liu-example -o $(EXAMPLE_DIR) -f txt \
				--override='calamr_corpus.name=adhoc,calamr_default.renderer=graphviz'"

# do not invoke directly--used by the align<corpus> targets
.PHONY:			_aligncorp
_aligncorp:
			@rm -rf $(EXAMPLE_DIR)
			@$(call loginfo,aligning $(CORP_CONF))
			@$(MAKE) $(PY_MAKE_ARGS) pyharn ARG="align --override \
				'calamr_corpus.name=$(CORP_CONF),calamr_default.renderer=graphviz,calamr_default.flow_graph_result_caching=preemptive' \
				--rendlevel $(REND_LEVEL) \
				-o $(EXAMPLE_DIR) $(EXTRA_ARGS)"

# align and generate graphs for the adhoc corpus
.PHONY:			alignadhoc
alignadhoc:
			@$(MAKE) $(PY_MAKE_ARGS) \
				CORP_CONF=adhoc REND_LEVEL=5 _aligncorp \
				EXTRA_ARGS="-i test-resources/tiny-corp.json"

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
			@rm -rf $(EXAMPLE_DIR)
			@$(MAKE) pyharn ARG="align -o $(EXAMPLE_DIR) -r 2 -f txt \
			    --override='calamr_corpus.name=adhoc,calamr_default.renderer=graphviz'"

# create examples of graphs in latex friendly EPS
.PHONY:			graphexampleseps
graphexampleseps:
			@rm -rf $(EXAMPLE_DIR)
			@$(MAKE) pyharn ARG="align -o $(EXAMPLE_DIR) -r 2 -f txt \
			    --override='calamr_corpus.name=adhoc,calamr_graph_render_graphviz.extension=eps,calamr_default.renderer=graphviz'"

# copy the graphs and guide to GitHub pages
.PHONY:			cpgraphs
cpgraphs:
			@$(call loginfo,copy graphs)
			mkdir -p $(PY_DOC_BUILD_HTML)
			cp -r doc $(PY_DOC_BUILD_HTML)


## Clean
#
# remove everything (careful)
.PHONY:			vaporizedep
vaporizedep:
			rm -fr corpus download
			git checkout corpus
