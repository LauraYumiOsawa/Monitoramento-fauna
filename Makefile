PYTHON ?= python3

.PHONY: install-runtime install-train install-modelo-b test info info-b validate train \
	arm64-smoke arm64-system sim-embedded modelo-b-prepare modelo-b-train modelo-b-pipeline

install-runtime:
	$(PYTHON) -m pip install -e .

install-train:
	$(PYTHON) -m pip install -e ".[train]"

install-modelo-b:
	$(PYTHON) -m pip install -e ".[modelo-b]"

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

info:
	PYTHONPATH=src $(PYTHON) -m animal_monitor info

info-b:
	PYTHONPATH=src $(PYTHON) -m animal_monitor_b info

validate:
	animal-monitor validate-dataset --data config/dataset.yaml

train:
	animal-monitor train --config config/train.yaml

arm64-smoke:
	./scripts/qemu_arm64_smoke.sh

arm64-system:
	./scripts/qemu_system_arm64.sh

sim-embedded:
	./scripts/docker_resource_sim.sh

# --- Modelo B: dois estagios (deteccao de presenca + classificacao de especie) ---
modelo-b-prepare:
	animal-monitor-b prepare-data --data config/dataset.yaml

modelo-b-train:
	animal-monitor-b train-detector --config config/modelo_b/detector.yaml
	animal-monitor-b train-classifier --config config/modelo_b/classifier.yaml

modelo-b-pipeline:
	./scripts/modelo_b_pipeline.sh
