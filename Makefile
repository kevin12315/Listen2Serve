PY ?= python
export PYTHONPATH := src:scripts

.PHONY: install test lint check-data smoke

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts

check-data:
	$(PY) scripts/check_dataset.py

# 离线冒烟：本地桩服务当裁判端点，跑完 网关→量规→判定 全链路，零外部调用
smoke:
	$(PY) scripts/smoke_offline.py
