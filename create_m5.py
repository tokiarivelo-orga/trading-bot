import os

def create_m5():
    src_path = "backend/src/strategies/generated/xauusd_snd_qm_structure_adaptive_m1_v1.py"
    dst_path = "backend/src/strategies/generated/xauusd_snd_qm_structure_adaptive_m5_v1.py"

    with open(src_path, "r") as f:
        content = f.read()

    # Replacements
    content = content.replace("class XauusdSndQmStructureAdaptiveM1:", "class XauusdSndQmStructureAdaptiveM5:")
    content = content.replace('name="xauusd_snd_qm_structure_adaptive_m1",', 'name="xauusd_snd_qm_structure_adaptive_m5",')
    content = content.replace('entry_timeframe="M1",', 'entry_timeframe="M5",')
    content = content.replace('confirmation_timeframes=("M15",),', 'confirmation_timeframes=("H1",),')
    
    content = content.replace('"zone_tf_minutes": 5,', '"zone_tf_minutes": 15,')
    content = content.replace('"entry_tf_minutes": 1,', '"entry_tf_minutes": 5,')
    content = content.replace('"htf_key": "M15",', '"htf_key": "H1",')

    with open(dst_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    create_m5()
