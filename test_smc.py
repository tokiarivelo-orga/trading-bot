import sys
from pathlib import Path
sys.path.insert(0, str(Path("backend").resolve()))
from src.strategies.generated.smc_dl_m5_v1 import SmcDlM5V1
import traceback

try:
    s = SmcDlM5V1()
    print("Success")
except Exception as e:
    traceback.print_exc()
