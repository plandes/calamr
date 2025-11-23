# Multiple corpus example

There are two example corpora: `corp-a` and `corp-b`.  Each have a
configuration file in the [config](./config) directory and corpus JSON file in
the [corpus](./corpus) directory.

This example shows how to use the command line to create adhoc corpora (which
is any corpus that isn't the AMR release, Little Prince or AMR Bio corpora).


There are two source files:

* `harness.sh`: life-cycle for AMR corpora given JSON source/summaries and
  shows how to use the command line to create new corpora
* `harness.py`: example of how to programmatically access and align the corpora


To run the example, first create the AMR corpora:

1. Create `corp-a` by parsing the sentences in the `corpus/corp-a.json` into
   AMR file `corpus/corp-a.txt`: `./harness.sh mkcorp corp-b`
2. Do the same for `corp-b`: `./harness.sh mkcorp corp-b`
3. Optionally align via the command line: `./harness.sh align corp-a`


Now that the `corpus/corp-a.txt` AMR output is available, run the examples:

1. Access and align the corpora in the example Python file
   [harness.py](./harness.py).
2. Optionally remove all corpus artifacts: `./harness.sh clean corp-a`
