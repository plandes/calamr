#!/bin/bash
# @meta {desc: 'Integration test script', date: '2024-03-15'}
#
# Integration testing for CALAMR.  This script uses the command line tool to
# download corpora, align documents, score AMR graphs along with several other
# tests.  The proxy corpus tests need the AMR 3.0 corpus file installed at
# ~/Desktop/amr_annotation_3.0_LDC2020T02.tgz.


PROG=$(basename $0)
PYTHON_BIN="python"
CALAMR_BIN="${PYTHON_BIN} ./calamr"
unset CALAMRRC


## Utiility
#
function prhead() {
    echo "--------------------${1}:"
}

function bail() {
    msg=$1 ; shift
    echo "$PROG: error: $msg"
    echo "environment:"
    export
    exit 1
}

function report() {
    msg=$1 ; shift
    echo "$PROG: test outcome: $msg"
}

function assert_success() {
    ret=$1 ; shift
    if [ $ret -ne 0 ] ; then
	bail "last command failed"
    fi
}

function assert_files() {
    dir=$1 ; shift
    expect=$1 ; shift
    if [ ! -d $dir ] ; then
	bail "no directory: $dir"
    fi
    cnt=$(ls $dir | wc -l)
    if [ $cnt -ne $expect ] ; then
	bail "expected $expect files in $dir but got $cnt"
    fi
    report "file count $dir: ${cnt}...OK"
}

function assert_file_len() {
    file=$1 ; shift
    expect=$1 ; shift
    cnt=$(cat $file | wc -l)
    if [ ! -f $file ] ; then
	bail "no file: $file"
    fi
    if [ $cnt -ne $expect ] ; then
	bail "expected $file to have $expect lines but got $cnt"
    fi
    report "file length $file: ${cnt}...OK"
}

function assert_output_len() {
    cmd=$1 ; shift
    expect=$1 ; shift
    cnt=$($cmd | wc -l)
    if [ $cnt -ne $expect ] ; then
	bail "expected <$cmd> to have $expect lines but got $cnt"
    fi
    report "command output <$cmd>: ${cnt}...OK"
}

## Setup
#
function setup_env() {
    # env vars here are set by framework and not configurable
    CONF_DIR=${HOME}
    APP_RC="${CONF_DIR}/.calamrrc"
    CACHE_DIR="${HOME}/.cache/calamr"
    LP_CORP="${CACHE_DIR}/corpus/amr-rel/amr-bank-struct-v3.0.txt"
    LP_CORP_OUT="./lp.txt"

    # AMR 3.0 corpus needs to be downloaded and put on the desktop
    amr_file=amr_annotation_3.0_LDC2020T02.tgz
    AMR_FILE_SRC=${HOME}/Desktop/${amr_file}
    AMR_FILE_DST=${CACHE_DIR}/download/${amr_file}
}

function check_amr_rel_corpus() {
    if [ ! -f ${AMR_FILE_SRC} ] ; then
	bail "missing AMR corpus file: $AMR_30_CORP"
    fi
}

function clean() {
    prhead "clean"

    make cleanall
    assert_success $?

    rm -fr ${LP_CORP_OUT} example scored.csv
    assert_success $?
}

function configure() {
    prhead "configure"

    echo "src/config/dot-calamrrc -> ${APP_RC}"
    cp src/config/dot-calamrrc ${APP_RC}
    assert_success $?
}


## Little Prince corpus and basic tests
#
function test_corpus_access() {
    prhead "test corpus access"

    assert_output_len "${CALAMR_BIN} keys --override calamr_corpus.name=little-prince" 1
    if [ -f $LP_CORP ] ; then
	echo "little prince corpus download: ${LP_CORP}...ok"
    else
	bail "missing little prince corpus: ${LP_CORP}"
    fi
}

function test_align() {
    prhead "alignment"

    ${CALAMR_BIN} mkadhoc --corpusfile corpus/micro/source.json
    assert_success $?

    rm -fr example
    ${CALAMR_BIN} aligncorp liu-example -f txt -o example
    assert_success $?

    assert_files "example/liu-example" 7
}

function test_scoring() {
    prhead "scoring"

    ${CALAMR_BIN} penman ${LP_CORP} -o ${LP_CORP_OUT} --limit 5 \
	     --override amr_default.parse_model=xfm_bart_base
    assert_success $?
    assert_file_len ${LP_CORP_OUT} 57

    ${CALAMR_BIN} score ${LP_CORP} --parsed ${LP_CORP_OUT} \
	   --methods calamr,smatch,wlk -o scored
    assert_success $?
    assert_file_len scored.csv 6
}


## Proxy tests
#
function test_proxy() {
    prhead "proxy-corpus"

    if [ ! -f ${AMR_FILE_DST} ] ; then
	mkdir $(dirname ${AMR_FILE_DST})
	assert_success $?

	cp ${AMR_FILE_SRC} ${AMR_FILE_DST}
	assert_success $?
    fi

    ${PYTHON_BIN} ./src/bin/merge-proxy-anons.py
    assert_success $?
    assert_output_len "${CALAMR_BIN} keys --override calamr_corpus.name=proxy-report" 367

    rm -fr example
    ${CALAMR_BIN} aligncorp 20041010_0024 -f txt -o example \
	   --override calamr_corpus.name=proxy-report
    assert_success $?
    assert_files example/20041010_0024 7
}


## Main
#
function main() {
    setup_env
    check_amr_rel_corpus
    clean
    configure
    test_corpus_access
    test_align
    test_scoring
    test_proxy
    clean
}

main
