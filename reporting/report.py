"""
Evidence-based report of one analysis run (all layers), exportable as HTML, Markdown or JSON.

Every stated result carries its provenance (KNOWN / ESTIMATED / ASSUMED / UNKNOWN), the
method, a confidence and the evidence behind it. Decoded content is reported as VERIFIED
only when a CRC validates it; otherwise the best hypothesis is shown as UNVERIFIED.
"""

import base64
import datetime as _dt
import html
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.evidence import Evidence, Parameter


def _num(v):
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        return float(f"{v:.6g}")
    return v


def _fmt(v) -> str:
    v = _num(v)
    if isinstance(v, float):
        return f"{v:,.6g}"
    return str(v)


def _ev(e: Evidence) -> Dict[str, Any]:
    return {"description": e.description, "source": e.source, "score": _num(float(e.score))}


def _param(p: Parameter) -> Dict[str, Any]:
    return {"name": p.name, "value": _fmt(p.value), "unit": p.unit, "source": p.source, "method": p.method or "",
            "confidence": None if p.confidence is None else _num(float(p.confidence)),
            "evidence": [_ev(e) for e in p.evidence]}


def build_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Collect everything the tabs show into one serialisable dict."""
    cfg = results.get("config") or {}
    params: Dict[str, Parameter] = results.get("parameters") or {}
    cls = results.get("classification")
    post = results.get("postdemod")
    demod = results.get("demod_result")
    buf = results.get("buffer")
    rep: Dict[str, Any] = {
        "title": "SIH 26147 - Automated Signal Analysis Report",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "input": {
            "file": cfg.get("file_path", ""),
            "file_name": os.path.basename(cfg.get("file_path", "")),
            "sample_type": str(cfg.get("dtype", "")),
            "sample_rate_hz": _num(buf.sample_rate) if buf is not None and buf.sample_rate else None,
            "sample_rate_source": {
                "duration": "computed from the user-entered recording duration (KNOWN)",
                "blind": "ASSUMED: blind hypothesis (measured samples/symbol x a standard symbol rate)",
                "unknown": "UNKNOWN: results in normalised units",
                "manual": "entered by the user (KNOWN)",
                "header": "WAV header (KNOWN)"}.get(cfg.get("sample_rate_source"), "file header / metadata"),
            "samples": int(len(buf.samples)) if buf is not None else 0,
            "notes": list(results.get("capture_notes") or []),
        },
        "detection": {"signals_of_interest": len(results.get("sois") or [])},
        "classification": None,
        "parameters": [_param(p) for p in params.values()],
        "demodulation": None,
        "bitstream": None,
        "deinterleaving": None,
        "fec": None,
        "framing": None,
        "result": None,
        "limitations": [
            "FEC identification covers a library of convolutional codes (K 3-9, rate 1/2, 1/3, punctured "
            "2/3, 3/4, 5/6), Reed-Solomon GF(2^8) profiles (p = 0x11D, fcr = 0), Hamming codes, LDPC codes with "
            "known parity-check matrices (IEEE 802.11 n=648 r3/4 and project-generated codes), blind LDPC "
            "parity-check recovery (needs more aligned error-free codewords than n) and RS + convolutional "
            "concatenated codes; turbo, BCH and polar codes are not identified.",
            "Pseudo-random interleavers: the LFSR address-generator family is searched and a rank-deficiency "
            "period without a structured interleaver is reported as a suspected PR interleaver; an arbitrary "
            "permutation is not reconstructed.",
            "Block interleavers larger than sqrt(N) bits are hypothesised from the burst length (a whole "
            "number of blocks) and accepted only when FEC + CRC validate.",
            "Convolutional (Forney) interleavers are searched for B <= 16 branches and M <= 16; diagonal "
            "interleavers over the same R x C hypotheses as block interleavers (both diagonal directions).",
            "A blind (ASSUMED) sample rate is a hypothesis: only samples/symbol is measured.",
            "Decoded content is VERIFIED only by a CRC; otherwise it is the best unverified hypothesis.",
        ],
    }
    if cls is not None:
        rep["classification"] = {"family": cls.family, "modulation": str(params["Modulation"].value)
                                 if "Modulation" in params else cls.modulation,
                                 "confidence": _num(float(cls.confidence)),
                                 "evidence": [_ev(e) if isinstance(e, Evidence) else {"description": str(e)}
                                              for e in (cls.evidence or [])][:20]}
    if demod is not None:
        rep["demodulation"] = {"bits": int(len(demod.bits)), "evm_or_error_pct": _num(float(demod.evm_pct)),
                               "symbols": int(len(demod.constellation_points))}
    if post is not None:
        bs = post.bitstream
        rep["bitstream"] = {
            "hypothesis": bs.hypothesis, "bits": bs.n_bits,
            "randomness": {k: _num(float(v)) for k, v in bs.randomness.items()},
            "periodicity": [[int(l), _num(float(c))] for l, c in bs.periodicity],
            "frame_length_estimate": bs.frame_length,
            "blind_convolutional_estimate": bs.blind_conv.description if bs.blind_conv else None,
            "sync_search": [{"word": s.name, "length": s.length, "offset": s.offset, "hamming": s.hamming,
                             "hard_corr": _num(s.hard_corr), "soft_corr": _num(s.soft_corr), "lrt": _num(s.lrt),
                             "glrt": _num(s.glrt), "threshold": s.threshold, "detected": s.detected,
                             "inverted": s.inverted, "occurrences": len(s.occurrences), "period": s.period,
                             "p_emulation": _num(s.p_emulation), "confidence": s.confidence} for s in bs.sync],
        }
        best = post.best
        di = post.deint_best
        rep["deinterleaving"] = {
            "decision": "none" if di.kind == "none" else f"{di.kind}: {di.params}, offset {di.offset}",
            "validated": bool(best and best.crc_frames and di.kind != "none"),
            "period_estimate": None if not post.period_estimate or not post.period_estimate.period else {
                "period": post.period_estimate.period, "offset": post.period_estimate.offset,
                "deficiency": _num(post.period_estimate.deficiency),
                "ks": None if post.period_estimate.ks is None else _num(post.period_estimate.ks)},
            "candidates": [dict(zip(["stream", "type", "parameters", "offset", "strength", "detail", "validated"], d.as_row()))
                           for d in post.deint_candidates[:20]],
            "hypotheses_tested": post.hypotheses_tested,
        }
        rep["fec"] = {
            "family_probabilities": {k: _num(v) for k, v in post.family.probabilities.items()},
            "family": post.family.family,
            "decision": best.fec_name if best else "none",
            "description": best.fec_detail if best else "",
            "evidence": [_ev(e) for e in (post.family.evidence + (best.evidence if best else []))],
            "candidates": [dict(zip(["stream", "family", "code", "setting", "metric", "result", "crc_frames"], f.as_row()))
                           for f in post.fec_candidates[:25]],
        }
        fa = best.frames if best else None
        if fa is not None:
            rep["framing"] = {
                "layout": fa.layout.describe(fa.sync.length) if fa.layout and fa.sync else None,
                "sync": fa.sync.name if fa.sync else None,
                "frames": [{"index": f.index, "start_bit": f.start_bit, "length": f.length,
                            "crc": f.crc_name, "crc_ok": f.crc_ok, "text": f.text,
                            "hex": f.payload.hex(" ").upper()} for f in fa.frames],
                "valid_frames": fa.valid_frames, "layouts_tested": fa.layouts_tested,
                "evidence": [_ev(e) for e in fa.evidence],
            }
        rep["search_log"] = list(post.log)
        rep["search_seconds"] = _num(post.seconds)
    img = results.get("image")
    if img is not None:
        rep["recovered_file"] = {
            "format": img.format, "bytes": len(img.payload), "width": img.width, "height": img.height,
            "channels": img.channels, "crc_ok": img.crc_ok, "crc_received": f"0x{img.crc_received:08X}",
            "crc_computed": f"0x{img.crc_computed:08X}", "fec": img.fec, "hypothesis": img.hypothesis,
            "sync_bit": img.sync_bit, "sync_errors": img.sync_errors,
            "evidence": [_ev(e) for e in img.evidence],
        }
    sigs = results.get("signals") or []
    if len(sigs) > 1:                                   # capture with several transmissions
        rows = []
        for i, r in enumerate(sigs):
            b, im = r.get("band"), r.get("image")
            pr = r.get("parameters") or {}
            rows.append({"index": i + 1, "center_hz": _num(b.center_hz) if b else None,
                         "bandwidth_hz": _num(b.bandwidth_hz) if b else None, "snr_db": _num(b.snr_db) if b else None,
                         "modulation": str(pr["Modulation"].value) if "Modulation" in pr else None,
                         "result": (f"error: {r['error']}" if r.get("error") else im.describe() if im is not None
                                    else str(r.get("payload_text") or "")[:200]),
                         "crc_valid": bool(r.get("crc_valid"))})
        rep["signals"] = rows
    verified = bool(results.get("crc_valid"))
    conf = float(cls.confidence) if cls is not None else 0.0
    rep["result"] = {
        "payload_text": results.get("payload_text"),
        "status": "VERIFIED (CRC valid)" if verified else (
            "UNVERIFIED" if results.get("payload_text") else "NOT DECODED"),
        "processing_chain": _chain(rep),
        "overall_confidence": round(min(0.99, 0.5 + 0.49 * conf) if verified else 0.4 * conf, 2),
    }
    return rep


def _chain(rep) -> str:
    parts = []
    c = rep.get("classification")
    if c:
        parts.append(c["modulation"])
    d = rep.get("deinterleaving")
    if d and d["decision"] != "none":
        parts.append("de-interleave " + d["decision"])
    f = rep.get("fec")
    if f and f["decision"] not in ("none", None):
        parts.append(f["decision"])
    fr = rep.get("framing")
    if fr and fr.get("layout"):
        parts.append(fr["layout"])
    return " -> ".join(parts) if parts else "-"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def to_json(rep: Dict[str, Any]) -> str:
    return json.dumps(rep, indent=2, default=str)


def to_markdown(rep: Dict[str, Any]) -> str:
    L: List[str] = [f"# {rep['title']}", "", f"Generated {rep['generated']}", ""]
    r = rep["result"]
    L += ["## Result", "", f"* **Status:** {r['status']}", f"* **Decoded payload:** `{r['payload_text']}`",
          f"* **Processing chain:** {r['processing_chain']}", f"* **Overall confidence:** {r['overall_confidence']}", ""]
    if rep.get("signals"):
        L += ["## Signals in the capture", "", "| # | Centre (Hz) | Bandwidth (Hz) | SNR (dB) | Modulation | Result |",
              "|---|---|---|---|---|---|"]
        L += [f"| {x['index']} | {x['center_hz']} | {x['bandwidth_hz']} | {x['snr_db']} | {x['modulation']} | "
              f"{x['result']} |" for x in rep["signals"]] + [""]
    i = rep["input"]
    L += ["## Input", "", f"* File: `{i['file_name']}` ({i['samples']:,} complex samples)",
          f"* Sample type: {i['sample_type'] or '-'}",
          f"* Sample rate: {_fmt(i['sample_rate_hz'])} Hz ({i['sample_rate_source']})"]
    L += [f"* {n}" for n in i["notes"]] + [""]
    c = rep.get("classification")
    if c:
        L += ["## Modulation", "", f"* {c['family']} / **{c['modulation']}** (confidence {c['confidence']})"]
        L += [f"  * {e['description']}" for e in c["evidence"][:10]] + [""]
    L += ["## Parameters", "", "| Parameter | Value | Unit | Source | Method | Confidence |", "|---|---|---|---|---|---|"]
    for p in rep["parameters"]:
        L.append(f"| {p['name']} | {p['value']} | {p['unit']} | {p['source']} | {p['method']} | "
                 f"{'' if p['confidence'] is None else p['confidence']} |")
    L.append("")
    b = rep.get("bitstream")
    if b:
        L += ["## Bitstream (L8)", "", f"* {b['bits']} bits ({b['hypothesis']}); randomness "
              + ", ".join(f"{k} {v}" for k, v in b["randomness"].items()),
              f"* Frame length estimate: {b['frame_length_estimate']}",
              f"* Blind convolutional estimate: {b['blind_convolutional_estimate'] or 'none'}", "",
              "| Sync word | Offset | Hamming | Hard corr | LRT | GLRT | Threshold | Detected |", "|---|---|---|---|---|---|---|---|"]
        for s in b["sync_search"]:
            L.append(f"| {s['word']} | {s['offset']} | {s['hamming']}/{s['length']} | {s['hard_corr']} | {s['lrt']} | "
                     f"{s['glrt']} | {s['threshold']} | {s['confidence'] if s['detected'] else 'no'} |")
        L.append("")
    d = rep.get("deinterleaving")
    if d:
        L += ["## De-interleaving (L9)", "", f"* Decision: **{d['decision']}** "
              f"({'validated by FEC + CRC' if d['validated'] else 'not validated'}); {d['hypotheses_tested']} "
              f"hypotheses tested"]
        if d["period_estimate"]:
            pe = d["period_estimate"]
            L.append(f"* Rank-deficiency period estimate: P = {pe['period']} at offset {pe['offset']} "
                     f"(deficiency {pe['deficiency']}, K-S {pe['ks']})")
        L.append("")
    f = rep.get("fec")
    if f:
        L += ["## FEC (L10)", "", f"* Decision: **{f['decision']}** - {f['description']}",
              "* Family probabilities: " + ", ".join(f"{k} {v:.2f}" for k, v in f["family_probabilities"].items())]
        L += [f"  * {e['description']}" for e in f["evidence"]] + [""]
    fr = rep.get("framing")
    if fr:
        L += ["## Framing (L11)", "", f"* Layout: {fr['layout'] or 'not validated'}",
              f"* CRC-valid frames: {fr['valid_frames']}/{len(fr['frames'])}", "",
              "| # | Start bit | Length | CRC | Valid | Text |", "|---|---|---|---|---|---|"]
        for x in fr["frames"]:
            L.append(f"| {x['index']} | {x['start_bit']} | {x['length']} | {x['crc']} | {x['crc_ok']} | "
                     f"`{x['text']}` |")
        L.append("")
    L += ["## Limitations", ""] + [f"* {s}" for s in rep["limitations"]]
    return "\n".join(L) + "\n"


def to_html(rep: Dict[str, Any], figures: Optional[Sequence[Tuple[str, bytes, int]]] = None) -> str:
    """figures: optional (caption, PNG bytes, display width px) plots, embedded as data URIs so the
    HTML file stays self-contained."""
    e = html.escape
    css = ("body{font-family:Segoe UI,Arial,sans-serif;margin:18px;color:#1d2433;background:#fff}"
           "h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:18px 0 6px;border-bottom:1px solid #d0d7e2;"
           "padding-bottom:3px;color:#1f497d}table{border-collapse:collapse;margin:6px 0;font-size:12px}"
           "td,th{border:1px solid #d0d7e2;padding:3px 7px;text-align:left;vertical-align:top}"
           "th{background:#eef2f8}.ok{color:#1a7f37;font-weight:bold}.bad{color:#b42318;font-weight:bold}"
           ".mono{font-family:Consolas,monospace}.box{background:#f5f7fb;border:1px solid #d0d7e2;padding:8px 10px}"
           "ul{margin:4px 0 4px 18px;padding:0}li{margin:2px 0;font-size:12px}"
           ".fig{display:inline-block;margin:6px 14px 6px 0;vertical-align:top}.cap{font-size:11px;color:#64748b}")
    r = rep["result"]
    ok = r["status"].startswith("VERIFIED")
    H = [f"<html><head><meta charset='utf-8'><title>{e(rep['title'])}</title><style>{css}</style></head><body>",
         f"<h1>{e(rep['title'])}</h1><div>Generated {e(rep['generated'])}</div>",
         "<h2>Result</h2><div class='box'>",
         f"<div>Status: <span class='{'ok' if ok else 'bad'}'>{e(r['status'])}</span></div>",
         f"<div>Decoded payload: <span class='mono'>{e(str(r['payload_text'])[:2000])}</span></div>",
         f"<div>Processing chain: {e(r['processing_chain'])}</div>",
         f"<div>Overall confidence: {r['overall_confidence']}</div></div>"]
    if figures:
        H.append("<h2>Plots</h2><table style='border:none'><tr>")
        for cap, png, width in figures:
            src = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
            H.append(f"<td style='border:none;padding:4px 14px 4px 0'><div class='fig'><img src='{src}' width='{width}'>"
                     f"<div class='cap'>{e(cap)}</div></div></td>")
        H.append("</tr></table>")
    if rep.get("signals"):
        H += ["<h2>Signals in the capture</h2><table><tr><th>#</th><th>Centre (Hz)</th><th>Bandwidth (Hz)</th>"
              "<th>SNR (dB)</th><th>Modulation</th><th>Result</th></tr>"]
        H += [f"<tr><td>{x['index']}</td><td>{e(str(x['center_hz']))}</td><td>{e(str(x['bandwidth_hz']))}</td>"
              f"<td>{e(str(x['snr_db']))}</td><td>{e(str(x['modulation']))}</td>"
              f"<td class='{'ok' if x['crc_valid'] else ''}'>{e(x['result'])}</td></tr>" for x in rep["signals"]]
        H.append("</table>")
    i = rep["input"]
    H += ["<h2>Input</h2><ul>", f"<li>File: <span class='mono'>{e(i['file_name'])}</span> "
          f"({i['samples']:,} complex samples), sample type {e(i['sample_type'] or '-')}</li>",
          f"<li>Sample rate {e(_fmt(i['sample_rate_hz']))} Hz - {e(i['sample_rate_source'])}</li>"]
    H += [f"<li>{e(n)}</li>" for n in i["notes"]] + ["</ul>"]
    c = rep.get("classification")
    if c:
        H += [f"<h2>Modulation</h2><div>{e(c['family'])} / <b>{e(c['modulation'])}</b> "
              f"(confidence {c['confidence']})</div><ul>"]
        H += [f"<li>{e(x['description'])}</li>" for x in c["evidence"][:10]] + ["</ul>"]
    H.append("<h2>Parameters (provenance)</h2><table><tr><th>Parameter</th><th>Value</th><th>Unit</th>"
             "<th>Source</th><th>Method</th><th>Confidence</th></tr>")
    for p in rep["parameters"]:
        H.append(f"<tr><td>{e(p['name'])}</td><td class='mono'>{e(p['value'])}</td><td>{e(p['unit'])}</td>"
                 f"<td>{e(p['source'])}</td><td>{e(p['method'])}</td>"
                 f"<td>{'' if p['confidence'] is None else p['confidence']}</td></tr>")
    H.append("</table>")
    b = rep.get("bitstream")
    if b:
        H += [f"<h2>Bitstream (L8)</h2><div>{b['bits']} bits ({e(b['hypothesis'])}); frame length estimate "
              f"{b['frame_length_estimate']}; blind convolutional estimate: "
              f"{e(b['blind_convolutional_estimate'] or 'none')}</div>",
              "<table><tr><th>Sync word</th><th>Offset</th><th>Hamming</th><th>Hard corr</th><th>Soft corr</th>"
              "<th>LRT</th><th>GLRT</th><th>Threshold</th><th>Occ.</th><th>P(emulation)</th><th>Detected</th></tr>"]
        for s in b["sync_search"]:
            H.append(f"<tr><td>{e(s['word'])}</td><td>{s['offset']}</td><td>{s['hamming']}/{s['length']}</td>"
                     f"<td>{s['hard_corr']}</td><td>{s['soft_corr']}</td><td>{s['lrt']}</td><td>{s['glrt']}</td>"
                     f"<td>{s['threshold']}</td><td>{s['occurrences']}</td><td>{s['p_emulation']:.2e}</td>"
                     f"<td class='{'ok' if s['detected'] else ''}'>{s['confidence'] if s['detected'] else 'no'}"
                     f"</td></tr>")
        H.append("</table>")
    d = rep.get("deinterleaving")
    if d:
        H += [f"<h2>De-interleaving (L9)</h2><div>Decision: <b>{e(d['decision'])}</b> "
              f"<span class='{'ok' if d['validated'] else ''}'>"
              f"{'validated by FEC + CRC' if d['validated'] else '(not validated)'}</span>; "
              f"{d['hypotheses_tested']} hypotheses tested</div>"]
        if d["period_estimate"]:
            pe = d["period_estimate"]
            H.append(f"<div>Rank-deficiency period estimate: P = {pe['period']} at offset {pe['offset']} "
                     f"(deficiency {pe['deficiency']}, K-S {pe['ks']})</div>")
        if d["candidates"]:
            H.append("<table><tr><th>Bit stream</th><th>Type</th><th>Parameters</th><th>Offset</th><th>Strength</th><th>Detail</th>"
                     "<th>Validated</th></tr>")
            for x in d["candidates"]:
                H.append("<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in x.values()) + "</tr>")
            H.append("</table>")
    f = rep.get("fec")
    if f:
        H += [f"<h2>FEC (L10)</h2><div>Decision: <b>{e(f['decision'])}</b> - {e(f['description'])}</div>",
              "<table><tr><th>FEC family</th><th>Probability</th></tr>"]
        H += [f"<tr><td>{e(k)}</td><td>{v:.2f}</td></tr>" for k, v in f["family_probabilities"].items()]
        H += ["</table><ul>"] + [f"<li>{e(x['description'])}</li>" for x in f["evidence"]] + ["</ul>"]
        if f["candidates"]:
            H.append("<table><tr><th>Bit stream</th><th>Family</th><th>Code</th><th>Setting</th><th>Metric</th><th>Result</th>"
                     "<th>CRC frames</th></tr>")
            for x in f["candidates"]:
                H.append("<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in x.values()) + "</tr>")
            H.append("</table>")
    fr = rep.get("framing")
    if fr:
        H += [f"<h2>Framing (L11)</h2><div>Layout: {e(fr['layout'] or 'not validated')}; CRC-valid frames "
              f"{fr['valid_frames']}/{len(fr['frames'])} ({fr['layouts_tested']} layouts tested)</div>",
              "<table><tr><th>#</th><th>Start bit</th><th>Length</th><th>CRC</th><th>Valid</th><th>Text</th></tr>"]
        for x in fr["frames"]:
            H.append(f"<tr><td>{x['index']}</td><td>{x['start_bit']}</td><td>{x['length']}</td><td>{e(str(x['crc']))}</td>"
                     f"<td class='{'ok' if x['crc_ok'] else 'bad'}'>{x['crc_ok']}</td>"
                     f"<td class='mono'>{e(x['text'])}</td></tr>")
        H.append("</table>")
    H += ["<h2>Limitations</h2><ul>"] + [f"<li>{e(s)}</li>" for s in rep["limitations"]] + ["</ul></body></html>"]
    return "\n".join(H)


def save(rep: Dict[str, Any], path: str, figures: Optional[Sequence[Tuple[str, bytes, int]]] = None) -> str:
    ext = os.path.splitext(path)[1].lower()
    text = to_html(rep, figures) if ext in (".html", ".htm") else to_markdown(rep) if ext == ".md" else to_json(rep)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path
