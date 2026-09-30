"""Compare QAM edge-symbol policies (off / ls / guarded) on the legacy QAM captures (.iq + .wav)."""
import sys, time, glob, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from ingestion.capture import load_capture
from modulation.unified_classifier import classify_capture
from demodulation.qam_receiver import receive_qam, QAMReceiverConfig
TXT = 'BRIGHT VIXENS JUMP; DOZY FOWL QUACK'
files = []
for stem in ['Test-Data/all_modulations_output/16QAM', 'Test-Data/all_modulations_output/64QAM', 'Test-Data/all_modulations_output/256QAM',
             'Test-Data/qam_pangram_all_files/16qam_bright_vixens', 'Test-Data/qam_pangram_all_files/64qam_bright_vixens', 'Test-Data/qam_pangram_all_files/256qam_bright_vixens']:
    files += [stem + '.iq', stem + '.wav']
tot = {m: 0 for m in ('off', 'ls', 'guarded')}
for f in files:
    c = load_capture(f); cls = classify_capture(c.samples, c.sample_rate)
    row = []
    for m in ('off', 'ls', 'guarded'):
        r = receive_qam(c.samples, c.sample_rate, QAMReceiverConfig(modulation=cls.modulation, edge_mode=m), front=cls.front, classification=cls.linear)
        t = r.payload_text
        errs = sum(a != b for a, b in zip(t, TXT)) + abs(len(t) - len(TXT))
        tot[m] += errs; row.append(f'{m}:{errs}')
    print(f'{f[10:]:48s} {cls.modulation:7s}', ' '.join(row), flush=True)
print('TOTAL char errors', tot)
