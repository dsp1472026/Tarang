Image decoding tests (SIH 26147)
================================

Signals made with the signal path of Image Decoding/c.py (Space Downlink Simulator): frame
PREAMBLE 0xAA x4 | SYNC 1ACFFC1D | LEN u32 | PAYLOAD (W,H,C header + raw RGB or PNG) | CRC-32(payload),
RRC 0.35, IF offset 0.25 x fs. Noise added only where an SNR is listed.

Per test:  <name>.wav / .iq    the signal (the .iq has a .json sidecar with its 1 MHz sample rate)
           <name>_encoded.png  the image the transmitter sent
           <name>_recovered.png  the image the application recovered from the signal
           summary_sent_vs_recovered.png  all tests side by side

test                   modulation image        payload  file      SNR    result
demo_qpsk_raw          QPSK       256x191     raw      wav_iq    -      bit-exact, CRC valid, 172.9s
demo_8psk_png          8PSK       512x383     PNG      wav_iq    -      bit-exact, CRC valid, 68.6s
orig_16qam_raw         16QAM      256x165     raw      wav_iq    30 dB  bit-exact, CRC valid, 401.7s
orig_qpsk_png1024      QPSK       1024x662     PNG      wav_iq    -      bit-exact, CRC valid, 225.3s
pattern_64qam_png      64QAM      256x160     PNG      wav_iq    -      bit-exact, CRC valid, 281.5s
pattern_bpsk_raw       BPSK       96x60      raw      wav_real  25 dB  bit-exact, CRC valid, 36.2s
demo_2fsk_png          2FSK       200x149     PNG      wav_iq    -      bit-exact, CRC valid, 271.2s
orig_4fsk_png          4FSK       160x103     PNG      wav_iq    20 dB  bit-exact, CRC valid, 146.7s
pattern_256qam_raw     256QAM     96x60      raw      wav_iq    -      bit-exact, CRC valid, 124.0s
orig_qpsk_raw_noisy    QPSK       256x165     raw      iq        14 dB  bit-exact, CRC valid, 151.9s

Screenshot_qpsk_recovered.png: the original 'Screenshot ... _qpsk.wav' (256x165, QPSK), recovered bit-exact
with a valid CRC; _x4 is a Lanczos enlargement for viewing (no added detail - the transmitter sent 256 px;
orig_qpsk_png1024 shows the same slide sent at 1024 px).
