# Unified classification report

Blind FSK / PSK / QAM classification over every Test-Data capture (`.iq` and `.wav`). Truth comes from the sidecars and is used only for scoring.

| Input | Truth | Files | Classified correctly |
|---|---|---|---|
| .iq | 16PSK | 15 | 15/15 |
| .iq | 16QAM | 4 | 4/4 |
| .iq | 256QAM | 4 | 4/4 |
| .iq | 2FSK | 8 | 8/8 |
| .iq | 4FSK | 10 | 10/10 |
| .iq | 4GFSK | 2 | 2/2 |
| .iq | 64QAM | 4 | 4/4 |
| .iq | 8FSK | 2 | 2/2 |
| .iq | 8PSK | 22 | 22/22 |
| .iq | BPSK | 24 | 24/24 |
| .iq | GFSK | 1 | 1/1 |
| .iq | GMSK | 3 | 3/3 |
| .iq | MSK | 4 | 4/4 |
| .iq | QPSK | 25 | 25/25 |
| wav | 16PSK | 15 | 14/15 |
| wav | 16QAM | 4 | 4/4 |
| wav | 256QAM | 4 | 4/4 |
| wav | 2FSK | 2 | 2/2 |
| wav | 4FSK | 2 | 2/2 |
| wav | 64QAM | 4 | 4/4 |
| wav | 8FSK | 1 | 1/1 |
| wav | 8PSK | 16 | 16/16 |
| wav | BPSK | 18 | 18/18 |
| wav | QPSK | 19 | 19/19 |

**Total: 212/213 classified correctly** (family 213/213).

## Failures

| File | Truth | Classified | Payload |
|---|---|---|---|
| `Test-Data/16psk_4800baud_10db.wav` | 16PSK | PSK/8PSK | None |
