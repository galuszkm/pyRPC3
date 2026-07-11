# https://just.systems/man/en/

# SETTINGS

set dotenv-load := true
set shell := ["bash", "-cu"]

# VARIABLES

PACKAGE := "pyRPC3"
SOURCES := "src"
TESTS := "tests"

# DEFAULTS

# display help information
default:
    @just --list

# IMPORTS

import 'tasks/check.just'
import 'tasks/clean.just'
import 'tasks/commit.just'
import 'tasks/format.just'
import 'tasks/install.just'
import 'tasks/secrets.just'
import 'tasks/test.just'
