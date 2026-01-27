SHELL := /bin/bash

PY ?= python
RAW_DIR ?= data/raw
PROCESSED_DIR ?= data/processed

.PHONY: pipeline download prepare lookup clean serve

pipeline: download prepare lookup

requirements:
	$(PY) -m pip install -r requirements.txt

$(RAW_DIR):
	mkdir -p $(RAW_DIR)

$(PROCESSED_DIR):
	mkdir -p $(PROCESSED_DIR)

download: $(RAW_DIR)
	$(PY) scripts/download_arsuppgjor.py --raw-dir $(RAW_DIR)

prepare: $(PROCESSED_DIR)
	$(PY) scripts/prepare_arsuppgjor.py --raw-dir $(RAW_DIR) --processed-dir $(PROCESSED_DIR)

lookup: $(PROCESSED_DIR)
	$(PY) scripts/lookup_vm_entities.py --parquet $(PROCESSED_DIR)/arsuppgjor_combined.parquet

serve:
	$(PY) app.py

clean:
	rm -rf $(RAW_DIR) $(PROCESSED_DIR)
