PROJECT_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: install test build-manifests verify-stage1

install:
	cd $(PROJECT_ROOT) && pip install -e ".[dev]"

test:
	cd $(PROJECT_ROOT) && pytest

build-manifests:
	cd $(PROJECT_ROOT) && pc-build-manifests

verify-stage1:
	cd $(PROJECT_ROOT) && pc-verify-stage1
