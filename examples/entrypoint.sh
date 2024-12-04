#!/bin/sh

binary_name=$1
shift

cargo run --bin=$binary_name --features=$binary_name -- $@