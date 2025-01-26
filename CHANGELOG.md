# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


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
[Unreleased]: https://github.com/plandes/calamr/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/plandes/calamr/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/plandes/calamr/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/plandes/calamr/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/plandes/calamr/compare/v0.0.0...v0.0.1

[zensols.propbank]: https://github.com/plandes/propbankdb
[zensols.amr]: https://github.com/plandes/amr
