PYTHON ?= python3

.PHONY: install-runtime install-train install-modelo-b test test-data test-report test-matriz test-tudo \
	info info-b validate train \
	arm64-smoke arm64-system sim-embedded modelo-b-prepare modelo-b-train modelo-b-pipeline

install-runtime:
	$(PYTHON) -m pip install -e .

install-train:
	$(PYTHON) -m pip install -e ".[train]"

install-modelo-b:
	$(PYTHON) -m pip install -e ".[modelo-b]"

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

test-data:
	PYTHONPATH=src $(PYTHON) scripts/testar_tudo.py

# Roda todos os testes e gera outputs/testes/relatorio.html (mesmo se algum teste falhar).
test-report:
	PYTHONPATH=src $(PYTHON) scripts/testar_tudo.py; status=$$?; \
	$(PYTHON) scripts/relatorio_testes_html.py; exit $$status

# Matriz das 25 versões: QEMU ARM64 + Docker com limites de CPU/RAM (~5 min). Atualiza o relatório.
test-matriz:
	$(PYTHON) scripts/testar_matriz.py; status=$$?; \
	$(PYTHON) scripts/relatorio_testes_html.py; exit $$status

# Tudo: testes unitários + matriz + relatório.
test-tudo:
	PYTHONPATH=src $(PYTHON) scripts/testar_tudo.py; s1=$$?; \
	$(PYTHON) scripts/testar_matriz.py; s2=$$?; \
	$(PYTHON) scripts/relatorio_testes_html.py; \
	if [ $$s1 -ne 0 ] || [ $$s2 -ne 0 ]; then exit 1; fi

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
