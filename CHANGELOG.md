# Changelog

All notable changes to StochRoll will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Basic distribution statistics on `Roll`: variance, standard deviation,
  quantiles, and inclusive at-most probabilities.

## [0.1.0] - 2026-07-31

### Added

- Initial public release of StochRoll.
- Vectorized Monte Carlo simulation of dice rolls and dice pools.
- Core `Roller`, `Roll`, `Pool`, and `Event` APIs.
- Reproducible simulations through seeded random-number generation.
- Arithmetic, comparison, logical, and reduction operations on simulation results.
- Dice-pool operations for keeping, dropping, and rerolling dice.
- Structural selection, broadcasting, concatenation, stacking, and routing operations.
- Statistical methods for expected values and event probabilities.
- NumPy integration, static type annotations, documentation, and test coverage.

[Unreleased]: https://github.com/linkhack/stochroll/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/linkhack/stochroll/releases/tag/v0.1.0
