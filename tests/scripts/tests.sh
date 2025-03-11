#!/bin/zsh

# Run patterns
rm -rf tests/scripts/test_run_pat
qumin hydra.run.dir=tests/scripts/test_run_pat \
    data=tests/data/TestPackage/test.package.json \
    action=patterns

# Run entropies
rm -rf tests/scripts/test_run_ent
qumin hydra.run.dir=tests/scripts/test_run_ent \
    data=tests/data/TestPackage/test.package.json \
    action=H patterns=tests/scripts/test_run_pat/metadata.json

# Run lattices
rm -rf tests/scripts/test_run_lat
qumin hydra.run.dir=tests/scripts/test_run_lat \
    data=tests/data/TestPackage/test.package.json \
    action=lattice patterns=tests/scripts/test_run_pat/metadata.json

# Run macroclasses
rm -rf tests/scripts/test_run_mac
qumin hydra.run.dir=tests/scripts/test_run_mac \
    data=tests/data/TestPackage/test.package.json \
    action=macroclasses patterns=tests/scripts/test_run_pat/metadata.json

# Run class heatmap
rm -rf tests/scripts/test_run_hmap
qumin hydra.run.dir=tests/scripts/test_run_hmap \
    data=tests/data/TestPackage/test.package.json \
    action=heatmap patterns=tests/scripts/test_run_pat/metadata.json

# Run evaluation
rm -rf tests/scripts/test_run_eval
qumin hydra.run.dir=tests/scripts/test_run_eval \
    data=tests/data/TestPackage/test.package.json \
    action=eval
