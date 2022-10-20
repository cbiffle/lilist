#!/bin/bash

TOOLCHAIN="${TOOLCHAIN:-nightly}"

cargo +$TOOLCHAIN miri test
cargo +$TOOLCHAIN miri run --bin no_std
