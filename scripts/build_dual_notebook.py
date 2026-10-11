#!/usr/bin/env python3
"""Generate a hash-pinned notebook for bounded shared preprocessing/A/B studies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_notebook(output: Path, digest: str) -> None:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Full runtime SHA required.")
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": text.splitlines(True),
            }
        )

    md(
        "# Hai nhánh mô học A/B — Colab\n\nDefault chỉ preprocess, chưa train classifier. CPU modes: preprocess/cpu. T4 modes: gpu_smoke chạy hai lần độc lập rồi neural. Labels cấp ca; không gán cluster/noise/novelty thành nhãn ung thư patch. Dữ liệu nguồn giữ nguyên.\n"
    )
    code(
        'from pathlib import Path\n\nfrom google.colab import drive\n\ndrive.mount("/content/drive")\nROOT=Path("/content/drive/MyDrive/histology")\n'
    )
    code(
        'RUN_MODE="preprocess"\nSTUDY_ID="dual-20261011-v001"\nWORK_MINUTES=90\nRUNTIME_SHA256='
        + repr(digest)
        + '\nRUNTIME_ZIP=ROOT/"runtime/histology-dual-runtime.zip"\n'
    )
    code("""import hashlib
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone

if hashlib.sha256(RUNTIME_ZIP.read_bytes()).hexdigest()!=RUNTIME_SHA256:raise ValueError("Runtime SHA mismatch")
CODE=Path("/content/histology-dual-runtime")/RUNTIME_SHA256[:16];CODE.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(RUNTIME_ZIP) as archive:
 for name in archive.namelist():
  p=Path(name)
  if p.is_absolute() or ".." in p.parts:raise ValueError("Unsafe runtime path")
 archive.extractall(CODE)
subprocess.run([sys.executable,"-m","pip","install","--quiet","scikit-learn==1.8.0"],check=True)
sys.path.insert(0,str(CODE))
from histology_data.mil_runtime import stage_training_inputs
from histology_data.mil_case_cache import prepare_case_cache
STOP=datetime.now(timezone.utc)+timedelta(minutes=WORK_MINUTES)
def deadline():return time.monotonic()+max(0,(STOP-datetime.now(timezone.utc)).total_seconds())
BUNDLE_ID="6e412d65e429ad19c3bcec645a19d7ee06fe75e560ef67149233f14d4175b0ff"
stage=stage_training_inputs(ROOT/"bundles"/BUNDLE_ID,ROOT/"features/v001",Path("/content/dual-stage"),deadline())
cache=prepare_case_cache(stage["bundle_path"],stage["feature_root"],Path("/content/dual-stage/case-caches"),deadline())
setup={k:str(v) if isinstance(v,Path) else v for k,v in stage.items()};setup["case_cache"]=str(cache)
SETUP=Path("/content/dual-source-setup.json");SETUP.write_text(json.dumps(setup))
""")
    code("""CONFIG=Path("/content/dual-config.json")
config={"study_id":STUDY_ID,"cohort":"common16","setup_path":str(SETUP),"output_root":str(ROOT/"clustering_dual/v001"/STUDY_ID),"source_code_sha256":RUNTIME_SHA256,"runtime_archive_path":str(RUNTIME_ZIP),"recover_stale_locks":False}
CONFIG.write_text(json.dumps(config))
if RUN_MODE not in {"preprocess","cpu","gpu_smoke","neural"}:raise ValueError("Unknown explicit mode")
result=subprocess.run([sys.executable,"-m","histology_data.dual_cli","--mode",RUN_MODE,"--config",str(CONFIG),"--deadline-utc",STOP.isoformat()],cwd=CODE,check=False)
if result.returncode:raise RuntimeError("Run failed; preserve artifacts and inspect status/logs.")
""")
    md(
        "## Resume và kết thúc\n\nGiữ STUDY_ID/runtime SHA/settings để resume. Chỉ bật recover_stale_locks sau khi xác minh owner cũ đã dừng; cross-host cần evidence đúng token/host/boot/session cho từng lock. Complete của CPU là cpu_complete; B1 và báo cáo vẫn có stages riêng. Không auto-click/SSH/keep-alive giả. Sau workers dừng và receipts verified, flush Drive và dừng đúng runtime.\n"
    )
    code("""# Chỉ chạy cell này sau khi workers đã dừng và artifacts đã được kiểm.
# from google.colab import drive
# drive.flush_and_unmount(timeout_ms=60000)
# Dừng runtime qua Colab/colab-manager; không tắt máy cá nhân.
""")
    expanded = []
    for cell in cells:
        text = "".join(cell["source"])
        marker = "from histology_data.mil_runtime import stage_training_inputs"
        if cell["cell_type"] == "code" and marker in text:
            before, after = text.split(marker, 1)
            first = dict(cell); first["source"] = before.splitlines(True)
            second = dict(cell)
            after = after.replace("\nfrom histology_data.mil_case_cache import prepare_case_cache", "", 1)
            second["source"] = ("from histology_data.mil_case_cache import prepare_case_cache\n" + marker + "\n" + after).splitlines(True)
            expanded.extend([first, second])
        else:
            expanded.append(cell)
    cells = expanded
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256((str(index) + "".join(cell["source"])).encode()).hexdigest()[:12]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            ensure_ascii=False,
            indent=1,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build_notebook(a.output, a.sha256)
