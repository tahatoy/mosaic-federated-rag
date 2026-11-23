# MOSAIC — common commands. Usage: make <target>
PY ?= python3

.PHONY: help install install-branch install-exp certs hq branch1 branch2 test experiments tr-mmlu

help:
	@echo "make install        HQ dependencies"
	@echo "make install-branch branch-node dependencies"
	@echo "make install-exp    evaluation dependencies"
	@echo "make certs          generate CA and server certificates (HQ_LAN_IP=<ip>)"
	@echo "make hq             start the headquarters node"
	@echo "make branch1|2      start a branch node"
	@echo "make test           run the pytest suite"
	@echo "make experiments    run the original hypothesis experiments (H1-H4)"
	@echo "make tr-mmlu        run the TR-MMLU benchmark (E10)"

install:        ; $(PY) -m pip install -r requirements.txt
install-branch: ; $(PY) -m pip install -r requirements-branch.txt
install-exp:    ; $(PY) -m pip install -r requirements-experiments.txt
certs:          ; bash backend/scripts/generate_certs.sh
hq:             ; bash scripts/start_hq.sh
branch1:        ; bash scripts/start_vm1.sh
branch2:        ; bash scripts/start_vm2.sh
test:           ; $(PY) -m pytest backend/tests -v
experiments:    ; bash scripts/experiments/run_all_experiments.sh
tr-mmlu:        ; bash scripts/experiments/q1/q1_15_tr_mmlu.sh
