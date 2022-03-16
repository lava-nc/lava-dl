
# Release 0.2.0

The [lava-dl](https://github.com/lava-nc/lava-dl) library now supports automated generation of Lava processes for a trained network described by hdf5 network configuration using our Network Exchange (NetX) library.

## New Features and Improvements

* Released Network Exchange (NetX) library to support automated creation of Lava process for a deep network. We support hdf5 network exchange format. Support for more formats will be introduced in future. ([PR #30](https://github.com/lava-nc/lava-dl/pull/30), [Issue #29](https://github.com/lava-nc/lava-dl/issues/29))

## Bug Fixes and Other Changes
- Fixed bug with pre-hook quantization function on conv blocks ([PR#13](https://github.com/lava-nc/lava-dl/pull/13))

## Breaking Changes
- No breaking changes in this release

## Known Issues
- Issue training with GPU for lava-dl-slayer on Windows machine.

## What's Changed
* Create PULL_REQUEST_TEMPLATE.md & ISSUE_TEMPLATE.md by @mgkwill in https://github.com/lava-nc/lava-dl/pull/27
* Hardware neuron parameters exchange and fixed precision instruction precision compatibility by @bamsumit in https://github.com/lava-nc/lava-dl/pull/25
* Pilotnet link fix by @bamsumit in https://github.com/lava-nc/lava-dl/pull/31
* Bugfix: CUBA neuron normalization applied to current state by @bamsumit in https://github.com/lava-nc/lava-dl/pull/35
* Netx by @bamsumit in https://github.com/lava-nc/lava-dl/pull/30
* Streamline PilotNet SNN notebook with RefPorts by @bamsumit in https://github.com/lava-nc/lava-dl/pull/37
* Fix for failing tests/lava/lib/dl/netx/test_hdf5.py by @bamsumit in https://github.com/lava-nc/lava-dl/pull/44
* Update ci-build.yml by @mgkwill in https://github.com/lava-nc/lava-dl/pull/42
* Install by @mgkwill in https://github.com/lava-nc/lava-dl/pull/45


**Full Changelog**: https://github.com/lava-nc/lava-dl/compare/v0.1.1...v0.2.0