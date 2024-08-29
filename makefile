## makefile automates the build and deployment for python projects


## Build system
#
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-cli python-doc python-doc-deploy markdown
CLEAN_DEPS +=		pycleancache
ADD_CLEAN_ALL +=	data corpus/micro/amr.txt
PY_DOC_BUILD_HTML_DEPS += cpgraphs


## Project
#
ENTRY =			./calamr
ALIGN_DIR =		results/align


## Includes
#
include ./zenbuild/main.mk


## Configure
#
# install dependencies needed for scoring AMRs (i.e. WLK)
.PHONY:			scoredeps
scoredeps:
			$(PIP_BIN) install $(PIP_ARGS) -r $(PY_SRC)/requirements-score.txt

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
			$(ENTRY) mkadhoc --override calamr_corpus.name=adhoc
			mkdir -p download
			( cat corpus/micro/amr.txt | bzip2 > download/micro.txt.bz2 )

## Alignment
#
# create adhoc corpus graphs for one example
.PHONY:			aligncorp
aligncorp:
			rm -rf example
			$(ENTRY) aligncorp liu-example -o example -f txt \
			    --override='calamr_corpus.name=adhoc,calamr_default.renderer=graphviz'

# do not invoke directly--used by the align<corpus> targets
.PHONY:			aligncorpall
aligncorpall:
			@echo "aligning $(CORP_CONF)"
			$(ENTRY) aligncorp ALL \
				--override \
				"calamr_corpus.name=$(CORP_CONF),calamr_default.renderer=graphviz,calamr_default.flow_graph_result_caching=preemptive" \
				--rendlevel $(REND_LEVEL) --cached \
				-o $(ALIGN_DIR)/$(CORP_CONF)

# align and generate graphs for the adhoc corpus
.PHONY:			alignadhoc
alignadhoc:
			make CORP_CONF=adhoc REND_LEVEL=5 aligncorpall

# align and generate graphs for the proxyreport corpus
.PHONY:			alignproxy
alignproxy:
			make CORP_CONF=proxy-report REND_LEVEL=5 \
				aligncorpall > align-proxy.log 2>&1 &


## Example graphs
#
# create examples of html graphs
.PHONY:			graphexampleshtml
graphexampleshtml:
			rm -rf example
			$(ENTRY) aligncorp ALL -o example -r 2 -f txt \
			    --override='calamr_corpus.name=adhoc,calamr_default.renderer=graphviz'

# create examples of graphs in latex friendly EPS
.PHONY:			graphexampleseps
graphexampleseps:
			rm -rf example
			$(ENTRY) aligncorp ALL -o example -r 2 -f txt \
			    --override='calamr_corpus.name=adhoc,calamr_graph_render_graphviz.extension=eps,calamr_default.renderer=graphviz'

# copy the graphs and guide to GitHub pages
.PHONY:			cpgraphs
cpgraphs:
			@echo "copy graphs"
			mkdir -p $(PY_DOC_BUILD_HTML)
			cp -r doc $(PY_DOC_BUILD_HTML)


## Test
#
# test: unit and integration
.PHONY:			testall
testall:		test
			./test/inttest

# remove everything (careful)
.PHONY:			vaporize
vaporize:		cleanall
			rm -fr corpus download
			git checkout corpus

# remove everything and more (careful!)
.PHONY:			hellfire
hellfire:		vaporize
			rm -fr ~/.cache/calamr ~/.calamrrc
