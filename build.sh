#!/bin/bash

cargo build --release
mv target/release/make_monster_params ./
