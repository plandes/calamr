#!/bin/bash

DRYRUN=0
PROG=$(basename $0)
USAGE="\
usage: $PROG mkcorp [name]
       $PROG align [name]
       $PROG clean [name]

  mkcorp    create the corpus AMR file from the JSON file
  align     align the corpus and output the results
  clean     remove corups files"

# fail out of the program with an error message
function bail() {
    msg=$1 ; shift
    usage=$1 ; shift
    echo "$PROG: error: $msg" > /dev/stderr
    if [ "$usage" == 1 ] ; then
	printf "\n${USAGE}\n" > /dev/stderr
    fi
    exit 1
}

# execute a command and optionally show the command before running
function cmd() {
    cmd=$1 ; shift
    echo "executing: $cmd"
    [ $DRYRUN -eq 0 ] && eval $cmd
}

function config() {
    name=$1 ; shift
    echo "creating corpus..."
    conf="config/${name}.conf"
    if [ ! -f $conf ] ; then
	bail "no config file: $conf"
    fi
}

function mkcorp() {
    config $@
    cmd "calamr -c ${conf} mkadhoc"
}

function align() {
    config $@
    cmd "calamr -c ${conf} aligncorp ALL"
}

function clean() {
    config $@
    cmd "rm -rf data/${name}"
    cmd "rm -rf corpus/${name}.txt"
}

# entry point
function main() {
    action=$1 ; shift
    name=$1 ; shift
    if [ -z "$action" ] ; then
	action=show
    fi
    if [ -z "$name" ] ; then
	bail "missing name" 1
    fi
    case $action in
	mkcorp)
	    mkcorp $name
	    ;;
	align)
	    align $name
	    ;;
	clean)
	    clean $name
	    ;;
	-h|--help)
	    printf "${USAGE}\n"
	    ;;
	*)
	    printf "${USAGE}\n"
	    exit 1
	    ;;
    esac
}

main $@
