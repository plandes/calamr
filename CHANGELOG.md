# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


## [1.0.0] - 2025-12-02
Major API interface change to make ad hoc document parsing and aligning easier.

### Removed
- `Application.get_resource` (see changes).

### Changed
- `Application` method changed to `get_resources` to use a context manager and
  access to data as documents and alignments.  The `README.md` and the examples
  have been updated to use this new API.  The API has also been paired down to
  only what's needed from a client point of view.  However, the rest of the API
  remains and is still accessible for more sophisticated clients.


## [0.4.0] - 2025-11-23
### Removed
- Integration test using pre-Pixi configuration.

### Changed
- Move import of corpus configuration to `app.conf` from `pkg.conf` to allow
  external configuration overriding for adhoc corpora.

### Added
- Added [zensols.relpo] `envdist` configuration for (almost) Internet
  disconnected environment Conda installs.
- Multiple corpus example.


## [0.3.0] - 2025-11-13
### Changed
- The source aligned tokens `s_toks` and `t_toks` of the Pandas data frame
  output of the flow data has been changed to JSON forms of `[<token text>
  [character span]]`.
- Add support for Python 3.12.
- Switch build tools to [pixi].
- Upgraded dependencies:
  - [zensols.propbankdb] 0.3.0
  - [zensols.amr] 0.2.4


## [0.2.0] - 2025-01-26
### Removed
- Support for Python 3.10.

### Added
- Conda `src/python/environment.yml` file.

### Changed
- Upgrade to [zensols.propbank] version 0.2.0 and other dependent packages
  (including [zensols.amr]).
- Update URLs to test.
- Fix integration test (ISI website is now offline).
- Clean up application CLI source.


## [0.1.2] - 2024-07-07
### Changed
- Fix inconsistent scoring due to dirty cache hits in role edge assignment.


## [0.1.1] - 2024-07-03
### Added
- Easier facade level access to create and align ad hoc annotated (for source
  vs. summary) AMR graphs.
- Examples of the ad hoc alignment of AMR graphs.


## [0.1.0] - 2024-05-11
### Changed
- Upgraded to [zensols.propbank] 0.1.0.


## [0.0.1] - 2023-12-11
### Added
- Initial version.


<!-- links -->
[Unreleased]: https://github.com/plandes/calamr/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/plandes/calamr/compare/v0.4.0...v1.0.0
[0.4.0]: https://github.com/plandes/calamr/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/plandes/calamr/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/plandes/calamr/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/plandes/calamr/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/plandes/calamr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/plandes/calamr/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/plandes/calamr/compare/v0.0.0...v0.0.1

[zensols.propbank]: https://github.com/plandes/propbankdb
[zensols.amr]: https://github.com/plandes/amr
[zensols.relpo]: https://github.com/plandes/relpo
[pixi]: https://pixi.sh
