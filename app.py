"""Verity web interface.

    streamlit run app.py

Paste an AI-written answer and the sources it should agree with (or use the SciFact
science corpus); Verity splits the answer into claims, retrieves evidence, verifies each
claim and shows the result sentence by sentence.
"""
from __future__ import annotations

import copy
import hashlib
import html
import os
import re
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------ constants
LABELS = ["SUPPORTED", "REFUTED", "NEI"]
VERDICTS = ["SUPPORTED", "REFUTED", "NEI", "UNCERTAIN"]
NAMES = {"SUPPORTED": "Supported", "REFUTED": "Refuted",
         "NEI": "Not enough evidence", "UNCERTAIN": "Uncertain"}
SHORT = {"SUPPORTED": "Supported", "REFUTED": "Refuted", "NEI": "No evidence", "UNCERTAIN": "Uncertain"}
# a glyph per verdict, so nothing is conveyed by colour alone
GLYPH = {"SUPPORTED": "&#10003;", "REFUTED": "&#10005;", "NEI": "?", "UNCERTAIN": "!"}
# how bad a verdict is, for marking a sentence that holds several claims
SEVERITY = {"SUPPORTED": 0, "NEI": 1, "UNCERTAIN": 2, "REFUTED": 3}

EXAMPLES = {
    "Planets": {
        "answer": ("Jupiter is the largest planet in the Solar System. "
                   "Mars has two small moons, Phobos and Deimos. "
                   "Mars is the third planet from the Sun. "
                   "Saturn's rings are made mostly of water ice."),
        "sources": [
            ("jupiter.txt", "Jupiter is the largest planet in the Solar System. Its mass is more "
                            "than twice that of all the other planets combined."),
            ("mars.txt", "Mars is the fourth planet from the Sun. It has two small moons, "
                         "named Phobos and Deimos."),
        ],
    },
    "Water": {
        "answer": ("At standard atmospheric pressure, pure water boils at 100 degrees Celsius. "
                   "Pure water freezes at 10 degrees Celsius. "
                   "Water reaches its greatest density at about 4 degrees Celsius."),
        "sources": [
            ("water.txt", "At standard atmospheric pressure, pure water boils at 100 degrees "
                          "Celsius. Pure water freezes at 0 degrees Celsius. A water molecule is "
                          "made of two hydrogen atoms and one oxygen atom."),
        ],
    },
    "Marie Curie": {
        "answer": ("Marie Curie was the first woman to win a Nobel Prize. "
                   "Marie Curie won the Nobel Prize in Chemistry in 1911. "
                   "Marie Curie won only one Nobel Prize. "
                   "Marie Curie was born in Warsaw."),
        "sources": [
            ("curie.txt", "Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize "
                          "in Chemistry in 1911. She was the first woman to win a Nobel Prize, and "
                          "the first person to win Nobel Prizes in two different sciences."),
        ],
    },
    "Smoking (SciFact)": {
        "answer": ("Smoking is associated with an increased risk of lung cancer. "
                   "Vitamin D supplementation prevents all fractures in older adults. "
                   "Aspirin reduces the risk of colorectal cancer."),
        "scifact": True,
    },
}

FIGURES = [("Retrieval", "fig1_retrieval.png"), ("Verifiers", "fig2_verifiers.png"),
           ("Calibration", "fig3_calibration.png"), ("End to end", "fig4_end_to_end.png"),
           ("RAGTruth", "fig5_ragtruth.png"), ("Laptop vs cloud", "fig6_scale.png")]
SRC_PASTE, SRC_SCIFACT = "Pasted sources", "SciFact corpus"
MAX_SOURCES = 4
FINETUNED_DIR = ROOT / "models" / "verity-deberta-scifact"

# ------------------------------------------------------------------ report component assets
# Colour tokens per theme. Verdict text colours are tuned for >= 4.5:1 on the card
# backgrounds of their theme; fills (bars, icons) carry dark ink in both themes.
REPORT_CSS = r"""
:root { color-scheme: dark;
  --bg:#0b0a0d; --panel:#131016; --panel2:#1a161d; --line:#2c262d; --text:#ddd8d4; --muted:#aaa1a4;
  --gold:#f0b54a; --gold-t:#f3c06a; --crim:#e0313b; --crim-t:#ff7a80; --ink:#0a0c0f;
  --ok:#22c55e; --ok-t:#4ade80; --bad:#fb7185; --bad-t:#fda4af; --nei:#94a3b8; --nei-t:#cbd5e1;
  --unc:#a78bfa; --unc-t:#c4b5fd; --mark:rgba(240,181,74,.24); --mark-line:rgba(240,181,74,.75);
  --lift:rgba(240,181,74,.07); --shadow:0 10px 30px -14px rgba(0,0,0,.8); }
[data-theme="light"] { color-scheme: light;
  --bg:#ffffff; --panel:#fbf9f7; --panel2:#f3eee8; --line:#e2d9d1; --text:#2a2326; --muted:#5d5357;
  --gold:#d4951f; --gold-t:#8a4b08; --crim:#c81e2a; --crim-t:#b4161f; --ink:#0a0c0f;
  --ok:#22c55e; --ok-t:#166534; --bad:#fb7185; --bad-t:#be123c; --nei:#94a3b8; --nei-t:#475569;
  --unc:#a78bfa; --unc-t:#6d28d9; --mark:rgba(240,181,74,.38); --mark-line:rgba(180,110,10,.8);
  --lift:rgba(212,149,31,.08); --shadow:0 10px 28px -16px rgba(60,40,20,.35); }
.v-SUPPORTED { --c:var(--ok); --ct:var(--ok-t); } .v-REFUTED { --c:var(--bad); --ct:var(--bad-t); }
.v-NEI { --c:var(--nei); --ct:var(--nei-t); } .v-UNCERTAIN { --c:var(--unc); --ct:var(--unc-t); }
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); }
body { font: 16px/1.5 "Source Sans 3", "Source Sans Pro", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased; overflow: hidden; }
#vr { padding: 2px 2px 6px; }
h2, h3, p { margin: 0; }
button { font: inherit; color: inherit; }
:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }
.ico { display: inline-flex; align-items: center; justify-content: center; flex: none; width: 1.1rem; height: 1.1rem;
  border-radius: 50%; font-size: .68rem; font-weight: 800; line-height: 1; background: var(--c); color: var(--ink); }
.chip { display: inline-flex; align-items: center; gap: .38rem; padding: .14rem .62rem .14rem .28rem; border-radius: 999px;
  font-size: .8rem; font-weight: 650; white-space: nowrap; color: var(--ct);
  background: color-mix(in srgb, var(--c) 13%, transparent); border: 1px solid color-mix(in srgb, var(--c) 50%, transparent); }

/* ---------- header */
.hl { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: .3rem 1rem; }
.hl h2 { font-size: 1.42rem; line-height: 1.25; font-weight: 700; letter-spacing: -.005em; }
.sub { font-size: .86rem; color: var(--muted); }
.bar { display: flex; gap: 3px; height: 2.1rem; margin: .9rem 0 .8rem; border-radius: .5rem; overflow: hidden; }
.seg { flex-basis: 0; min-width: 2.9rem; display: flex; align-items: center; justify-content: center; gap: .35rem;
  background: var(--c); color: var(--ink); font-size: .82rem; font-weight: 750; white-space: nowrap; overflow: hidden;
  transform-origin: left center; }
.seg .ico { background: rgba(10,12,15,.16); color: var(--ink); }
.seg .segname { font-weight: 600; }
.filters { display: flex; flex-wrap: wrap; gap: .45rem; }
.f { display: inline-flex; align-items: center; gap: .4rem; min-height: 2.05rem; padding: .2rem .75rem .2rem .45rem;
  border-radius: 999px; cursor: pointer; font-size: .84rem; font-weight: 650;
  color: var(--ct); background: color-mix(in srgb, var(--c) 9%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 45%, transparent); transition: background .15s, box-shadow .15s; }
.f[data-f="ALL"] { padding-left: .8rem; --c: var(--gold); --ct: var(--text); }
.f .n { font-variant-numeric: tabular-nums; font-weight: 750; padding: 0 .38rem; border-radius: 999px; min-width: 1.4rem; text-align: center;
  background: color-mix(in srgb, var(--text) 10%, transparent); }
.f:hover:not(:disabled) { background: color-mix(in srgb, var(--c) 20%, transparent); }
.f[aria-pressed="true"] { background: color-mix(in srgb, var(--c) 24%, transparent); border-color: var(--c);
  box-shadow: inset 0 0 0 1px var(--c); }
.f[aria-pressed="true"]::after { content: "\2713"; font-size: .72rem; margin-left: -.1rem; }
.f:disabled { cursor: default; opacity: .62; border-style: dashed; }
.engine { display: flex; flex-wrap: wrap; align-items: baseline; row-gap: .1rem; margin-top: .9rem; padding-top: .6rem;
  border-top: 1px solid color-mix(in srgb, var(--gold) 24%, transparent);
  font: .74rem/1.6 "Source Code Pro", ui-monospace, Consolas, monospace; color: var(--muted); }
.engine b { color: var(--gold-t); font-weight: 600; letter-spacing: .12em; }
.engine span { white-space: nowrap; }
.engine i { font-style: normal; opacity: .6; padding: 0 .4rem; }

/* ---------- layout */
.grid { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); gap: 1.6rem; margin-top: 1.5rem; align-items: start; }
.panel { border: 1px solid color-mix(in srgb, var(--gold) 30%, var(--line)); border-radius: .75rem; background: var(--panel);
  padding: 1rem 1.15rem 1.05rem; box-shadow: var(--shadow); }
h3 { font-size: 1.05rem; font-weight: 650; letter-spacing: .01em; }
.hint, .key { font-size: .8rem; line-height: 1.45; color: var(--muted); }
.hint { margin: .25rem 0 .75rem; }
.key { margin-top: .85rem; }

/* ---------- answer */
.answer { font-size: 1.03rem; line-height: 2.05; overflow-wrap: anywhere; }
.s { cursor: pointer; border-radius: 5px; outline-offset: 1px; }
.s .t { padding: .12rem .16rem; border-radius: 4px; box-decoration-break: clone; -webkit-box-decoration-break: clone;
  background: color-mix(in srgb, var(--c) 14%, transparent); border-bottom: 2px solid color-mix(in srgb, var(--c) 80%, transparent);
  transition: background .18s, box-shadow .18s, color .25s; }
.s.v-REFUTED .t { border-bottom-style: double; border-bottom-width: 3px; }
.s.v-NEI .t, .s.v-UNCERTAIN .t { border-bottom-style: dashed; }
.s:hover .t, .s.lit .t { background: color-mix(in srgb, var(--c) 30%, transparent);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--c) 70%, transparent); }
.s:focus-visible { outline: 2px solid var(--gold); }
.s.off .t { background: transparent; border-bottom-color: color-mix(in srgb, var(--text) 25%, transparent); }
.s.off .mark { filter: grayscale(1); }
.nw { white-space: nowrap; }
.s .t:not(.last) { border-top-right-radius: 0; border-bottom-right-radius: 0; padding-right: 0; }
.s > .t + .nw > .t { border-top-left-radius: 0; border-bottom-left-radius: 0; padding-left: 0; }
.mark { display: inline-flex; align-items: center; gap: .2rem; vertical-align: .12em; margin: 0 .16rem 0 .22rem;
  padding: 0 .4rem 0 .16rem; height: 1.2rem; border-radius: 999px; font-size: .7rem; font-weight: 750; line-height: 1;
  color: var(--ct); background: color-mix(in srgb, var(--c) 15%, transparent); border: 1px solid color-mix(in srgb, var(--c) 55%, transparent); }
.mark .ico { width: .88rem; height: .88rem; font-size: .56rem; }

/* ---------- cards */
.cards-h { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }
.status { font-size: .8rem; color: var(--muted); }
.olap { font-size: .78rem; line-height: 1.45; color: var(--muted); margin: .3rem 0 .8rem; }
mark { color: inherit; background: var(--mark); border-radius: 3px; padding: 0 .08em;
  box-shadow: inset 0 -1px 0 var(--mark-line); }
.card { position: relative; border: 1px solid var(--line); border-left: 4px solid var(--c); border-radius: .7rem;
  background: var(--panel); padding: .85rem 1rem .95rem; margin-bottom: .85rem; box-shadow: var(--shadow);
  transition: box-shadow .2s, border-color .2s, background .2s, transform .2s; scroll-margin: 1.2rem; }
.card[hidden] { display: none; }
.card.lit { border-color: color-mix(in srgb, var(--c) 70%, var(--line)); border-left-color: var(--c); background: var(--panel2);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--c) 55%, transparent), 0 0 26px -6px color-mix(in srgb, var(--c) 45%, transparent); }
.card:focus { outline: none; }
.card:focus-visible { outline: 2px solid var(--gold); outline-offset: 3px; }
.card.flash { animation: flash 1.1s ease-out; }
@keyframes flash { 0% { box-shadow: 0 0 0 3px var(--gold), 0 0 34px -4px var(--gold); } 100% { box-shadow: var(--shadow); } }
.s.flash .t { animation: sflash 1.1s ease-out; }
@keyframes sflash { 0% { box-shadow: 0 0 0 2px var(--gold); } 100% { box-shadow: 0 0 0 0 transparent; } }
.card-top { display: flex; align-items: center; justify-content: space-between; gap: .6rem; flex-wrap: wrap; }
.who { display: flex; align-items: center; gap: .55rem; }
.num { font: 600 .8rem "Source Code Pro", ui-monospace, Consolas, monospace; color: var(--muted); }
.locate { display: inline-flex; align-items: center; gap: .3rem; cursor: pointer; font-size: .78rem; font-weight: 600;
  color: var(--muted); background: transparent; border: 1px solid var(--line); border-radius: 999px; padding: .12rem .6rem; }
.locate:hover { color: var(--text); border-color: color-mix(in srgb, var(--gold) 60%, var(--line)); }
.claim { font-size: 1.03rem; font-weight: 600; line-height: 1.45; margin: .55rem 0 .7rem; }

/* confidence: three rows, the 90% cut-off drawn on each */
.conf { display: grid; gap: .28rem; }
.conf-h { display: flex; justify-content: space-between; gap: .6rem; font-size: .7rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: .1rem; }
.cutkey { display: inline-block; width: 2px; height: .75rem; margin-right: .35rem; vertical-align: -.1rem;
  background: var(--text); box-shadow: 0 0 0 1px var(--panel); }
.row { display: grid; grid-template-columns: 6.6rem minmax(0, 1fr) 2.7rem 4.1rem; align-items: center; gap: .55rem; font-size: .8rem; }
.lab { color: var(--muted); white-space: nowrap; }
.row.kept .lab { color: var(--ct); font-weight: 650; }
.track { position: relative; height: 8px; border-radius: 4px; background: color-mix(in srgb, var(--text) 11%, transparent); }
.fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; min-width: 2px;
  background: color-mix(in srgb, var(--c) 55%, transparent); transform-origin: left center; }
.row.kept .fill { background: var(--c); }
.cut { position: absolute; top: -3px; bottom: -3px; width: 2px; margin-left: -1px; background: var(--text); opacity: .85;
  box-shadow: 0 0 0 1px var(--panel); }
.val { text-align: right; font-variant-numeric: tabular-nums; font-weight: 650; }
.keep { font-size: .72rem; font-weight: 650; color: var(--ct); white-space: nowrap; }
.conf-foot { font-size: .82rem; color: var(--muted); margin-top: .45rem; }
.conf-foot b { color: var(--text); font-weight: 650; }

/* evidence */
.ev { margin-top: .8rem; padding: .6rem .8rem .65rem; border-radius: .5rem; background: color-mix(in srgb, var(--text) 4.5%, transparent);
  border-left: 3px solid var(--c); }
.ev-h { display: flex; justify-content: space-between; gap: .6rem; flex-wrap: wrap; font-size: .68rem; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin-bottom: .3rem; }
.ev-k { letter-spacing: .04em; text-transform: none; font-weight: 600; font-size: .74rem; }
.ev q { display: block; quotes: none; font-size: .93rem; line-height: 1.6; }
.ev-meta { font-size: .76rem; margin-top: .4rem; color: var(--muted); }
.ev-meta b { color: var(--text); font-weight: 650; }
details { margin-top: .6rem; }
summary { cursor: pointer; font-size: .82rem; font-weight: 600; color: var(--muted); border-radius: .3rem; padding: .15rem 0; width: max-content; max-width: 100%; }
summary:hover { color: var(--text); }
.none { padding: 1.2rem; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: .7rem; }

/* ---------- reveal: the answer arrives as crimson noise and settles into order */
.revealing .s:not(.settled) .t { color: var(--crim-t); background: color-mix(in srgb, var(--crim) 10%, transparent);
  border-bottom-color: color-mix(in srgb, var(--crim) 55%, transparent); border-bottom-style: solid;
  text-shadow: -1px 0 rgba(224,49,59,.55), 1px 0 rgba(240,181,74,.35); }
.revealing .s:not(.settled) .mark { opacity: 0; transform: scale(.6); }
.revealing .card { opacity: 0; transform: translateY(10px); border-left-color: var(--crim); }
.revealing .seg { transform: scaleX(0); }
.revealing .fill { transform: scaleX(0); }
.s.settled .t { color: inherit; text-shadow: none; transition: color .3s, background .3s, border-color .3s, text-shadow .3s; }
.s.settled .mark { opacity: 1; transform: none; transition: opacity .25s, transform .3s cubic-bezier(.3,1.6,.5,1); }
.card.in { opacity: 1; transform: none; border-left-color: var(--c);
  transition: opacity .35s ease-out, transform .35s ease-out, border-color .5s; }
.card.in .fill { transform: none; transition: transform .5s cubic-bezier(.2,.8,.2,1) .1s; }
.go .seg { transform: none; transition: transform .6s cubic-bezier(.2,.8,.2,1); }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* ---------- narrow */
@media (max-width: 760px) {
  .grid { grid-template-columns: minmax(0, 1fr); gap: 1.2rem; margin-top: 1.2rem; }
  .hl h2 { font-size: 1.18rem; }
  .seg .segname { display: none; }
  .panel { padding: .9rem .9rem 1rem; }
  .card { padding: .8rem .8rem .9rem; }
  .row { grid-template-columns: 5.4rem minmax(0, 1fr) 2.5rem 3.2rem; gap: .45rem; }
  .f { font-size: .8rem; padding-right: .6rem; }
}
"""

# Runs in <head>, before the body is parsed: copy the parent page's font faces (same
# origin, so the iframe matches the app's type) and decide whether to play the reveal.
REPORT_HEAD_JS = r"""
(function () {
  var d = document.documentElement;
  try {
    var css = [];
    for (var i = 0; i < parent.document.styleSheets.length; i++) {
      var ss = parent.document.styleSheets[i], rules;
      try { rules = ss.cssRules; } catch (e) { continue; }
      var base = ss.href || parent.location.href;
      for (var j = 0; j < rules.length; j++) {
        var r = rules[j];
        if (r.type !== 5 || !/Source (Sans|Code)/i.test(r.cssText)) continue;
        css.push(r.cssText.replace(/url\(["']?([^"')]+)["']?\)/g, function (m, u) {
          try { return 'url("' + new URL(u, base).href + '")'; } catch (e) { return m; } }));
      }
    }
    if (css.length) { var st = document.createElement('style'); st.textContent = css.join('\n'); document.head.appendChild(st); }
  } catch (e) {}
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches, seen = false;
  try { seen = parent.sessionStorage.getItem('verity-seen-__KEY__') === '1'; } catch (e) {}
  if (!reduce && !seen) d.classList.add('revealing');
})();
"""

REPORT_JS = r"""
(function () {
  var doc = document.documentElement, root = document.getElementById('vr'), KEY = root.dataset.key;
  var $$ = function (s, el) { return Array.prototype.slice.call((el || document).querySelectorAll(s)); };
  var sents = $$('.s'), cards = $$('.card'), chips = $$('.f'), status = document.getElementById('status');
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var store = null; try { store = parent.sessionStorage; } catch (e) {}

  /* ---- theme: follow the app's actual background, also after a theme switch */
  function syncTheme() {
    try {
      var el = parent.document.querySelector('[data-testid="stApp"]') || parent.document.body;
      var m = getComputedStyle(el).backgroundColor.match(/\d+(\.\d+)?/g);
      if (!m) return;
      var lum = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
      var t = lum > 0.5 ? 'light' : 'dark';
      if (doc.dataset.theme !== t) doc.dataset.theme = t;
      var ff = getComputedStyle(parent.document.body).fontFamily;
      if (ff && document.body.style.fontFamily !== ff) document.body.style.fontFamily = ff;
    } catch (e) {}
  }
  syncTheme(); setInterval(syncTheme, 1200);

  /* ---- size the iframe to its content */
  var lastH = 0;
  function fit() {
    if (typeof onScroll === 'function' && panel) onScroll();
    var h = Math.ceil(root.getBoundingClientRect().height) + 4;
    if (Math.abs(h - lastH) < 2) return;
    lastH = h;
    try {
      var fe = window.frameElement; if (!fe) return;
      fe.style.height = h + 'px'; fe.height = h;
      var box = fe.parentElement;                  /* Streamlit's element container keeps the first-guess height */
      if (box && box.getAttribute('data-testid') === 'stElementContainer') { box.style.height = h + 'px'; box.style.flex = '0 0 ' + h + 'px'; }
    } catch (e) {}
  }
  if (window.ResizeObserver) new ResizeObserver(fit).observe(root);
  window.addEventListener('resize', fit); fit();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);

  /* ---- keep the answer beside the cards while the page scrolls (wide layout only):
     the iframe never scrolls itself, so CSS sticky cannot work; follow the parent instead */
  var panel = document.querySelector('.answer-panel'), grid = document.querySelector('.grid'), queued = false;
  function stick() {
    queued = false;
    try {
      var fe = window.frameElement;
      if (!fe || window.innerWidth <= 760) { panel.style.transform = ''; return; }
      var top = 76, room = parent.innerHeight - top - 16, ph = panel.offsetHeight;
      if (ph > room) { panel.style.transform = ''; return; }
      var want = top - fe.getBoundingClientRect().top - grid.offsetTop;
      var y = Math.max(0, Math.min(want, grid.offsetHeight - ph));
      panel.style.transform = y ? 'translateY(' + Math.round(y) + 'px)' : '';
    } catch (e) {}
  }
  function onScroll() { if (!queued) { queued = true; requestAnimationFrame(stick); } }
  try {
    parent.addEventListener('scroll', onScroll, true); parent.addEventListener('resize', onScroll);
    window.addEventListener('pagehide', function () {
      try { parent.removeEventListener('scroll', onScroll, true); parent.removeEventListener('resize', onScroll); } catch (e) {}
    });
  } catch (e) {}
  stick();

  /* ---- linked highlighting between sentences and claim cards */
  function ids(el) { return (el.dataset.claims || el.dataset.claim || '').split(' ').filter(Boolean); }
  function card(i) { return document.getElementById('claim-' + i); }
  function sentenceOf(i) { return sents.filter(function (s) { return ids(s).indexOf(String(i)) >= 0; })[0]; }
  function light(list, on) {
    list.forEach(function (i) {
      var c = card(i); if (c) c.classList.toggle('lit', on);
      var s = sentenceOf(i); if (s) s.classList.toggle('lit', on);
    });
  }
  function flash(el) { el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); }
  function go(el) {
    if (!el) return;
    el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'center' });
    el.focus({ preventScroll: true }); flash(el);
  }
  sents.forEach(function (s) {
    var my = ids(s);
    ['mouseenter', 'focus'].forEach(function (ev) { s.addEventListener(ev, function () { light(my, true); }); });
    ['mouseleave', 'blur'].forEach(function (ev) { s.addEventListener(ev, function () { light(my, false); }); });
    function open() {
      var target = card(my[0]);
      if (target && target.hidden) setFilter('ALL');
      go(target);
    }
    s.addEventListener('click', open);
    s.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
    });
  });
  cards.forEach(function (c) {
    var my = [c.dataset.claim];
    c.addEventListener('mouseenter', function () { light(my, true); });
    c.addEventListener('mouseleave', function () { light(my, false); });
    c.addEventListener('focusin', function () { light(my, true); });
    c.addEventListener('focusout', function (e) { if (!c.contains(e.relatedTarget)) light(my, false); });
  });
  $$('.locate').forEach(function (b) {
    b.addEventListener('click', function () { go(sentenceOf(b.dataset.claim)); });
  });

  /* ---- filters */
  function setFilter(f) {
    var shown = 0;
    chips.forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.f === f ? 'true' : 'false'); });
    cards.forEach(function (c) { c.hidden = !(f === 'ALL' || c.dataset.v === f); if (!c.hidden) shown++; });
    sents.forEach(function (s) {
      s.classList.toggle('off', !ids(s).some(function (i) { var c = card(i); return c && !c.hidden; }));
    });
    document.getElementById('none').hidden = shown > 0;
    var label = f === 'ALL' ? 'all' : f === 'NEI' ? 'not enough evidence' : f.toLowerCase();
    status.textContent = f === 'ALL' ? 'Showing all ' + cards.length
      : 'Showing ' + shown + ' of ' + cards.length + ' (' + label + ')';
    try { store && store.setItem('verity-filter-' + KEY, f); } catch (e) {}
    fit();
  }
  chips.forEach(function (b) { b.addEventListener('click', function () { setFilter(b.dataset.f); }); });
  var saved = null; try { saved = store && store.getItem('verity-filter-' + KEY); } catch (e) {}
  if (saved && saved !== 'ALL' && chips.some(function (b) { return b.dataset.f === saved && !b.disabled; })) setFilter(saved);

  /* ---- reveal: crimson noise resolving, sentence by sentence, into the verdict view */
  if (doc.classList.contains('revealing')) {
    try { store && store.setItem('verity-seen-' + KEY, '1'); } catch (e) {}
    var answer = document.getElementById('answer'), noise = '#%&*+=<>/\\|{}[]?!01ABEFHKMNRSTVXZ';
    answer.setAttribute('aria-busy', 'true');
    /* a fresh report: bring it into view, so the reveal is seen rather than played off-screen */
    try {
      var fe = window.frameElement;
      if (fe && fe.getBoundingClientRect().top > parent.innerHeight * 0.6) {
        fe.style.scrollMarginTop = '132px'; fe.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } catch (e) {}
    var n = sents.length, stagger = Math.min(190, 600 / Math.max(1, n)), dur = 440, t0 = performance.now() + 120;
    var items = [];
    sents.forEach(function (s, k) {
      $$('.t', s).forEach(function (t) {
        items.push({ s: s, t: t, last: t.classList.contains('last'), text: t.textContent, at: k * stagger, done: false });
      });
    });
    requestAnimationFrame(function () { root.classList.add('go'); });
    function frame(now) {
      var el = now - t0, left = 0;
      items.forEach(function (it) {
        if (it.done) return;
        var p = Math.max(0, el - it.at) / dur;
        if (p >= 1) {
          it.t.textContent = it.text; it.done = true;
          if (it.last) {
            it.s.classList.add('settled');
            ids(it.s).forEach(function (i) { var c = card(i); if (c) c.classList.add('in'); });
          }
          return;
        }
        left++;
        var k = Math.max(0, Math.floor(p * it.text.length)), out = it.text.slice(0, k);
        for (var j = k; j < it.text.length; j++) {
          var ch = it.text[j];
          out += /\s/.test(ch) ? ch : noise[(Math.random() * noise.length) | 0];
        }
        it.t.textContent = out;
      });
      if (left) requestAnimationFrame(frame);
      else {
        cards.forEach(function (c) { c.classList.add('in'); });
        answer.removeAttribute('aria-busy');
        setTimeout(function () { doc.classList.remove('revealing'); root.dataset.revealMs = Math.round(performance.now() - t0 + 120); }, 320);
      }
    }
    requestAnimationFrame(frame);
  }
})();
"""

# ------------------------------------------------------------------ page + style
st.set_page_config(page_title="Verity - claim-by-claim answer checker",
                   page_icon=":material/fact_check:", layout="wide",
                   initial_sidebar_state="auto")

# Colours are mixed with `currentColor`, so the same rules give light text tints on the
# dark theme and dark tints on Streamlit's light theme (contrast holds in both).
CSS = """
<style>
/* Theme: an unchecked answer is chaos (crimson, fractured); checked against sources it
   becomes order (gold, geometric). Crimson and gold are decoration only and never appear
   on a verdict. Verdicts: green / rose / slate / violet, each with a glyph and a word. */
:root {
  --v-crimson: #e0313b; --v-gold: #f0b54a;
  --v-ok: #22c55e; --v-bad: #fb7185; --v-nei: #94a3b8; --v-unc: #a78bfa;
}
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 4.4rem; max-width: 1240px; }
.v-SUPPORTED { --c: var(--v-ok); } .v-REFUTED { --c: var(--v-bad); }
.v-NEI { --c: var(--v-nei); } .v-UNCERTAIN { --c: var(--v-unc); }
.v-muted { color: color-mix(in srgb, currentColor 72%, transparent); }
/* crimson haze at the top of the page (chaos), fading out before the results (order) */
[data-testid="stMain"] { background:
  radial-gradient(ellipse 820px 420px at 12% -6%, rgba(200,30,45,.17), transparent 70%),
  radial-gradient(ellipse 600px 360px at 92% 4%, rgba(200,30,45,.08), transparent 70%); }

/* ---------- header: chaos */
.v-hero { position: relative; padding: .4rem 0 1.1rem; overflow: hidden; }
.v-hero::before { content: ""; position: absolute; inset: 0; pointer-events: none;
  background: repeating-linear-gradient(0deg, rgba(224,49,59,.07) 0 1px, transparent 1px 4px);
  -webkit-mask-image: linear-gradient(90deg, #000 0%, rgba(0,0,0,.5) 45%, transparent 80%);
  mask-image: linear-gradient(90deg, #000 0%, rgba(0,0,0,.5) 45%, transparent 80%); }
.v-head { position: relative; display: flex; align-items: center; gap: 1.15rem; }
.v-core { flex: none; width: 48px; height: 54px; position: relative;
  clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
  background: linear-gradient(160deg, var(--v-crimson) 0%, #8f1d2a 42%, #b7791f 60%, var(--v-gold) 100%); }
.v-core::before { content: ""; position: absolute; inset: 2px;
  clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%); background: #0d0f13; }
.v-core::after { content: ""; position: absolute; left: 50%; top: 50%; width: 15px; height: 15px;
  transform: translate(-50%, -50%) rotate(45deg);
  background: linear-gradient(135deg, var(--v-crimson), var(--v-gold));
  box-shadow: 0 0 14px rgba(240,181,74,.55); animation: v-pulse 4s ease-in-out infinite; }
@keyframes v-pulse { 0%,100% { opacity: .78; } 50% { opacity: 1; } }
.v-title { font-size: 2.45rem; line-height: 1; font-weight: 800; letter-spacing: .26em; margin: 0; padding: 0;
  color: #c2410c;   /* solid fallback, readable on both themes */
  background: linear-gradient(90deg, color-mix(in srgb, var(--v-crimson) 85%, currentColor) 0%,
              color-mix(in srgb, #e8762e 85%, currentColor) 50%, color-mix(in srgb, var(--v-gold) 80%, currentColor) 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  filter: drop-shadow(-1px 0 0 rgba(224,49,59,.35)) drop-shadow(0 0 14px rgba(224,49,59,.22)); }
.v-tag { font-size: .74rem; letter-spacing: .34em; text-transform: uppercase; margin-top: .5rem;
  color: color-mix(in srgb, currentColor 75%, transparent); }
.v-tag b { font-weight: 600; color: color-mix(in srgb, var(--v-crimson) 55%, currentColor); }
.v-tag i { font-style: normal; font-weight: 600; color: color-mix(in srgb, var(--v-gold) 50%, currentColor); }
/* fractured edge under the header */
.v-fracture { height: 7px; margin: .9rem 0 0; opacity: .9;
  background: linear-gradient(90deg, var(--v-crimson), rgba(224,49,59,.35) 55%, transparent 90%);
  clip-path: polygon(0 40%, 4% 60%, 9% 20%, 15% 70%, 22% 35%, 27% 55%, 34% 25%, 41% 65%, 47% 40%, 55% 55%,
    63% 45%, 72% 52%, 82% 48%, 100% 50%, 100% 58%, 82% 56%, 72% 60%, 63% 53%, 55% 64%, 47% 50%, 41% 78%,
    34% 37%, 27% 68%, 22% 48%, 15% 85%, 9% 34%, 4% 75%, 0 56%); }
.v-lede { font-size: 1.02rem; margin: 1rem 0 1.2rem 0; max-width: 46rem; }

/* section labels: chaos steps crimson, order step gold */
.v-step { font-size: .72rem; font-weight: 700; letter-spacing: .2em; text-transform: uppercase; margin: .2rem 0 .5rem;
  color: color-mix(in srgb, var(--v-crimson) 58%, currentColor); }
.v-step span { color: color-mix(in srgb, currentColor 60%, transparent); font-weight: 600; margin-right: .5rem; }
.v-step.v-order { color: color-mix(in srgb, var(--v-gold) 52%, currentColor); display: flex; align-items: center; gap: .55rem; }
.v-hex { display: inline-block; width: .8rem; height: .9rem; background: var(--v-gold);
  clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%); }
.v-h { font-size: 1.05rem; font-weight: 650; margin: .2rem 0 .7rem; letter-spacing: .01em; }

/* primary button: crimson resolving to amber; white text passes contrast on both ends */
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg, #8e1622 0%, #a4321c 50%, #a45a0c 100%) !important; color: #fff !important;
  border: 1px solid rgba(240,181,74,.45) !important; letter-spacing: .06em; min-height: 2.9rem;
  box-shadow: 0 0 0 1px rgba(224,49,59,.12), 0 6px 22px -6px rgba(224,49,59,.45); }
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p { color: #fff !important; font-weight: 700; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  border-color: var(--v-gold) !important; box-shadow: 0 0 0 1px rgba(240,181,74,.3), 0 6px 26px -4px rgba(240,181,74,.4); }
button[kind="primary"]:focus-visible, [data-testid="stBaseButton-primary"]:focus-visible {
  outline: 2px solid var(--v-gold); outline-offset: 2px; }

/* panels */
.v-panel { border: 1px solid color-mix(in srgb, currentColor 16%, transparent); border-radius: .7rem;
  background: color-mix(in srgb, currentColor 3.5%, transparent); padding: 1rem 1.15rem; }
.v-order-panel { border-color: color-mix(in srgb, var(--v-gold) 30%, transparent); }
.v-empty { border-style: dashed; padding: 1.3rem 1.4rem; border-color: color-mix(in srgb, var(--v-gold) 35%, transparent); }
.v-empty p { margin: .2rem 0 .75rem; }

/* the merge: crimson resolving into gold, with a diamond at the seam */
.v-merge { position: relative; height: 2px; margin: 2rem 0 1.4rem; border: 0;
  background: linear-gradient(90deg, rgba(224,49,59,0) 0%, var(--v-crimson) 18%, #e8762e 50%, var(--v-gold) 82%, rgba(240,181,74,0) 100%); }
.v-merge::after { content: ""; position: absolute; left: 50%; top: 50%; width: 11px; height: 11px;
  transform: translate(-50%, -50%) rotate(45deg); background: var(--v-gold);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--v-gold) 18%, transparent), 0 0 14px rgba(240,181,74,.6); }

/* verdict chip */
.v-chip { display: inline-flex; align-items: center; gap: .38rem; padding: .14rem .6rem .14rem .3rem;
  border-radius: 999px; font-size: .78rem; font-weight: 650; letter-spacing: .02em; white-space: nowrap;
  color: color-mix(in srgb, var(--c) 52%, currentColor);
  background: color-mix(in srgb, var(--c) 14%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 50%, transparent); }
.v-ico { display: inline-flex; align-items: center; justify-content: center; width: 1.15rem; height: 1.15rem;
  border-radius: 50%; font-size: .72rem; font-weight: 800; background: var(--c); color: #0a0c0f; line-height: 1; }
.v-legend { display: flex; flex-wrap: wrap; gap: .45rem .55rem; }

/* checking: the scan line runs from crimson into gold */
.v-scan { position: relative; height: 3px; overflow: hidden; border-radius: 2px;
  background: linear-gradient(90deg, rgba(224,49,59,.18), rgba(240,181,74,.18)); margin: .6rem 0 .3rem; }
.v-scan::after { content: ""; position: absolute; top: 0; left: -35%; width: 35%; height: 100%;
  background: linear-gradient(90deg, transparent, var(--v-crimson), var(--v-gold), transparent);
  box-shadow: 0 0 10px rgba(240,181,74,.6); animation: v-sweep 1.6s ease-in-out infinite; }
@keyframes v-sweep { from { left: -35%; } to { left: 100%; } }
.v-scan-label { font-size: .78rem; letter-spacing: .18em; text-transform: uppercase;
  color: color-mix(in srgb, var(--v-crimson) 55%, currentColor); }
@media (prefers-reduced-motion: reduce) { .v-scan::after, .v-core::after { animation: none; } .v-scan::after { left: 32%; } }

/* sidebar: the pipeline runs from chaos (crimson) to order (gold) */
.v-side-h { font-size: .7rem; font-weight: 700; letter-spacing: .2em; text-transform: uppercase; margin: .4rem 0 .6rem;
  color: color-mix(in srgb, currentColor 78%, transparent); }
.v-steps { list-style: none; padding: 0; margin: 0 0 1.2rem; }
.v-steps li { position: relative; padding: 0 0 .85rem 2.2rem; margin: 0; font-size: .86rem; line-height: 1.45; }
.v-steps li::before { content: attr(data-n); position: absolute; left: 0; top: .05rem; width: 1.45rem; height: 1.45rem;
  display: flex; align-items: center; justify-content: center; font-size: .72rem; font-weight: 700;
  clip-path: polygon(50% 0, 100% 25%, 100% 75%, 50% 100%, 0 75%, 0 25%);
  background: color-mix(in srgb, var(--k) 22%, transparent); color: color-mix(in srgb, var(--k) 55%, currentColor); }
.v-steps li:not(:last-child)::after { content: ""; position: absolute; left: .72rem; top: 1.6rem; bottom: .15rem;
  width: 1px; background: linear-gradient(var(--k), var(--k2)); opacity: .45; }
.v-steps b { font-weight: 650; }
.v-defs { margin: 0 0 1.2rem; font-size: .84rem; line-height: 1.45; }
.v-defs div { margin-bottom: .6rem; }
.v-defs .v-chip { margin-bottom: .2rem; }
.v-limits { font-size: .8rem; line-height: 1.5; padding-left: 1.1rem; margin: 0;
  color: color-mix(in srgb, currentColor 78%, transparent); }
.v-limits li { margin-bottom: .35rem; }

@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding-top: 3.6rem; }
  .v-title { font-size: 1.9rem; letter-spacing: .2em; }
  .v-tag { letter-spacing: .2em; font-size: .66rem; }
  .v-core { width: 40px; height: 45px; }
  [data-testid="stButtonGroup"] button { padding-left: .6rem; padding-right: .6rem; }
}
</style>
"""
st.html(CSS)


# ------------------------------------------------------------------ helpers
def groq_key_available() -> bool:
    if os.environ.get("GROQ_API_KEY"):
        return True
    try:
        from dotenv import dotenv_values
        return bool(dotenv_values(ROOT / ".env").get("GROQ_API_KEY"))
    except ImportError:
        return False


@st.cache_resource(show_spinner=False, max_entries=8)
def paste_verity(sources: tuple[tuple[str, str], ...]):
    """One Verity per distinct source set. The NLI and embedding models inside are
    module-level caches in verity/, so they load once per process regardless."""
    from verity.pipeline import Verity
    return Verity(sources=list(sources))


@st.cache_resource(show_spinner=False)
def scifact_verity():
    from verity.pipeline import Verity
    return Verity.scifact()


@st.cache_resource(show_spinner=False)
def scifact_titles() -> dict[str, str]:
    from verity.data import scifact_corpus
    return {k: d.title for k, d in scifact_corpus().items()}


@st.cache_resource(show_spinner="First start: downloading the verifier, the SciFact corpus and its index ...")
def app_assets() -> None:
    """A fresh clone (or the live app's first start) lacks these; without the verifier
    the pipeline would silently fall back to the zero-shot model."""
    from verity.models import ensure_app_assets
    ensure_app_assets()


app_assets()


def esc(s) -> str:
    return html.escape(str(s))


def pct(p: float) -> str:
    if p >= 0.995:
        return "&gt;99%"
    if 0 < p < 0.005:
        return "&lt;1%"
    return f"{p:.0%}"


def chip(verdict: str, text: str | None = None) -> str:
    return (f'<span class="v-chip v-{verdict}"><span class="v-ico" aria-hidden="true">{GLYPH[verdict]}</span>'
            f'{esc(text or NAMES[verdict])}</span>')


def verifier_name(v: str) -> str:
    if Path(v).name == FINETUNED_DIR.name:
        return "fine-tuned DeBERTa-v3 (SciFact)"
    if v == "deberta-fever":
        return "zero-shot DeBERTa-v3 (MNLI+FEVER+ANLI)"
    return v


def source_label(src: str, scifact: bool) -> str:
    if not scifact:
        return f"<b>{esc(src)}</b>"
    title = scifact_titles().get(str(src), "")
    return f"<b>SciFact abstract {esc(src)}</b>" + (f" &middot; {esc(title)}" if title else "")


def theme_base() -> str:
    """The active theme ('dark' or 'light'); the component re-checks it in the browser."""
    try:
        t = st.context.theme.type
    except Exception:
        t = None
    return t if t in ("dark", "light") else (st.get_option("theme.base") or "dark")


def results_block() -> str:
    """The headline-results block of README.md (generated from results/), or ''."""
    try:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r"<!-- results:start -->(.*?)<!-- results:end -->", text, re.S)
    return m.group(1).strip() if m else ""


def headline(counts: dict, n: int) -> str:
    bad, unsure = counts["REFUTED"], counts["NEI"] + counts["UNCERTAIN"]
    if bad:
        return f"{bad} of {n} claim{'s' if n != 1 else ''} contradicted by the sources"
    if unsure:
        return f"No contradictions found, but {unsure} of {n} claim{'s' if n != 1 else ''} not settled"
    return f"All {n} claim{'s' if n != 1 else ''} supported by the sources"


# ------------------------------------------------------------------ report component
# The report is one self-contained HTML/CSS/JS document rendered in an iframe by
# components.html: no external scripts, fonts or URLs. The Python side builds all markup
# (so it reads correctly even before the script runs); the script only adds behaviour:
# linked highlighting, filters, the reveal, theme sync and sizing the iframe to its content.

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be been before being below
between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just may me might more
most much must my no nor not now of off on once only or other our ours out over own same she
should so some such than that the their theirs them then there these they this those through
to too under until up upon us very was we were what when where which while who whom why will
with would you your yours yet via per its one""".split())
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def _stem(w: str) -> str:
    """Lower-case and strip a few English endings, so 'moons' meets 'moon'."""
    w = w.lower()
    if w.endswith("'s"):
        w = w[:-2]
    for suf, keep in (("ing", 4), ("ies", 3), ("ed", 3), ("es", 3), ("s", 3)):
        if w.endswith(suf) and len(w) - len(suf) >= keep and not w.endswith("ss"):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def _content_words(text: str) -> set[str]:
    return {_stem(m.group()) for m in _WORD.finditer(text) if m.group().lower() not in STOPWORDS}


def overlap_marked(passage: str, claim: str) -> tuple[str, int]:
    """The passage, escaped, with every word it shares with the claim wrapped in <mark>
    (normalised word overlap, stopwords removed). Adjacent shared words share one mark.
    Returns the HTML and the number of distinct shared words."""
    want = _content_words(claim)
    out, pos, run_open, shared = [], 0, False, set()
    for m in _WORD.finditer(passage):
        w = m.group()
        hit = w.lower() not in STOPWORDS and _stem(w) in want
        gap = passage[pos:m.start()]
        if run_open and not (hit and gap.isspace()):
            out.append("</mark>")
            run_open = False
        out.append(esc(gap))
        if hit and not run_open:
            out.append("<mark>")
            run_open = True
        if hit:
            shared.add(_stem(w))
        out.append(esc(w))
        pos = m.end()
    if run_open:
        out.append("</mark>")
    out.append(esc(passage[pos:]))
    return "".join(out), len(shared)


def _ico(v: str) -> str:
    return f'<span class="ico" aria-hidden="true">{GLYPH[v]}</span>'


def _vchip(v: str) -> str:
    return f'<span class="chip v-{v}">{_ico(v)}{NAMES[v]}</span>'


def evidence_desc(e: dict) -> str:
    """How the evidence passages were formed, for the engine line."""
    sel, k = e.get("select"), e.get("top_k")
    if not e.get("scifact"):
        return "2-sentence windows of the sources"
    top = f"top {k} abstracts" if k else "closest abstracts"
    if sel == "windows":
        return f"{top}, every 2-sentence window"
    if sel:
        return f"{top}, {sel} closest sentences each"
    return f"{top}, whole abstracts"


def _engine(e: dict) -> str:
    if not e:
        return ""
    parts = [esc(e["mode"]), esc(verifier_name(e["verifier"])),
             esc(evidence_desc(e)),
             esc(f"T={e['temperature']:.2f}"), esc(f"q̂={e['qhat']:.3f} (90% target)"),
             esc(f"{e['seconds']:.1f} s")]
    return ('<div class="engine"><b>ENGINE</b>'
            + "".join(f'<span><i aria-hidden="true">/</i>{x}</span>' for x in parts) + "</div>")


def _glued(text: str, marks: str) -> str:
    """The sentence text with its last word and markers kept on one line, so a marker never
    wraps away from the sentence it belongs to."""
    k = text.rstrip().rfind(" ") + 1
    head = f'<span class="t">{esc(text[:k])}</span>' if k else ""
    return f'{head}<span class="nw"><span class="t last">{esc(text[k:])}</span>{marks}</span>'


def _answer_html(answer: str, claims) -> str:
    """The answer, each checked sentence a focusable button tinted by its worst verdict and
    followed by one marker (claim number + glyph) per claim it holds."""
    spans: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for i, c in enumerate(claims, 1):
        spans.setdefault((c.start, c.end), []).append((i, c.verdict))
    out, pos = [], 0
    for (a, e), items in sorted(spans.items()):
        if a < pos:                       # overlapping span: skip rather than duplicate text
            continue
        worst = max((v for _, v in items), key=SEVERITY.get)
        out.append(esc(answer[pos:a]))
        ids = " ".join(str(i) for i, _ in items)
        who = ", ".join(f"claim {i} {NAMES[v]}" for i, v in items)
        marks = "".join(f'<span class="mark v-{v}" aria-hidden="true">{_ico(v)}{i}</span>' for i, v in items)
        out.append(f'<span class="s v-{worst}" role="button" tabindex="0" data-claims="{ids}" '
                   f'aria-label="{esc(answer[a:e])} ({who}). Press Enter to open the claim.">'
                   f'{_glued(answer[a:e], marks)}</span>')
        pos = e
    out.append(esc(answer[pos:]))
    return "".join(out).replace("\n", "<br>")


def _confidence(c, qhat: float) -> str:
    """Three probability rows with the conformal cut-off drawn on each: every label whose
    probability reaches the line stays in the 90% label set."""
    cut = 1 - qhat
    rows = []
    for l in LABELS:
        p = c.probs.get(l, 0.0)
        kept = l in c.label_set
        rows.append(
            f'<div class="row v-{l}{" kept" if kept else ""}"><span class="lab">{SHORT[l]}</span>'
            f'<span class="track"><span class="fill" style="width:{100 * p:.1f}%"></span>'
            f'<span class="cut" style="left:{100 * cut:.1f}%"></span></span>'
            f'<span class="val">{pct(p)}</span>'
            f'<span class="keep">{"&#9670; in set" if kept else ""}</span></div>')
    aria = ("Verifier probabilities: " + ", ".join(f"{NAMES[l]} {c.probs.get(l, 0):.0%}" for l in LABELS)
            + f". Labels at or above {cut:.0%} stay in the 90% set: "
            + (", ".join(NAMES[l] for l in c.label_set) or "none") + ".")
    if c.verdict == "UNCERTAIN":
        open_ = ", ".join(NAMES[l] for l in c.label_set) if c.label_set else "none reaches the line"
        foot = f'Could not rule out: <b>{esc(open_)}</b>.'
    else:
        foot = f'Only <b>{NAMES[c.verdict]}</b> reaches the line, so it is the verdict.'
    return (f'<div class="conf" role="img" aria-label="{esc(aria)}">'
            f'<div class="conf-h"><span>Verifier probability</span>'
            f'<span><span class="cutkey" aria-hidden="true"></span>90% cut-off {pct(cut)}</span></div>'
            f'{"".join(rows)}</div><p class="conf-foot">{foot}</p>')


def _passage(ev, claim: str, label: str, scifact: bool) -> str:
    top = max(LABELS, key=lambda l: ev.probs.get(l, 0))
    body, k = overlap_marked(ev.text, claim)
    shared = f"{k} shared word{'s' if k != 1 else ''}" if k else "no shared words"
    return (f'<div class="ev v-{top}"><div class="ev-h"><span>{label}</span><span class="ev-k">{shared}</span></div>'
            f'<q>{body}</q><div class="ev-meta">{source_label(ev.source, scifact)} &middot; reads as '
            f'{NAMES[top].lower()} ({pct(ev.probs.get(top, 0))})</div></div>')


def _card(i: int, c, qhat: float, scifact: bool) -> str:
    v = c.verdict
    ev = ""
    if c.evidence:
        ev = _passage(c.evidence[0], c.claim, "Decisive passage", scifact)
        rest = c.evidence[1:]
        if rest:
            inner = "".join(_passage(e, c.claim, f"Also considered &middot; {j}", scifact)
                            for j, e in enumerate(rest, 2))
            ev += (f'<details><summary>{len(rest)} other passage{"s" if len(rest) > 1 else ""} '
                   f'considered</summary>{inner}</details>')
    return (f'<article class="card v-{v}" id="claim-{i}" data-claim="{i}" data-v="{v}" tabindex="-1" '
            f'aria-label="Claim {i}: {NAMES[v]}"><div class="card-top"><div class="who">'
            f'<span class="num">#{i}</span>{_vchip(v)}</div>'
            f'<button type="button" class="locate" data-claim="{i}" aria-label="Show claim {i} in the answer">'
            f'<span aria-hidden="true">&#8598;</span> In answer</button></div>'
            f'<p class="claim">{esc(c.claim)}</p>{_confidence(c, qhat)}{ev}</article>')


def report_component(report, eng: dict, theme: str) -> tuple[str, int]:
    """The full report document for components.html, and a first-guess height (the
    script resizes the iframe to its real content as soon as it runs)."""
    claims, counts, n = report.claims, report.counts, len(report.claims)
    scifact, qhat = bool(eng.get("scifact")), float(eng.get("qhat", 0.9))
    key = hashlib.sha1(report.to_json().encode()).hexdigest()[:16]
    segs = "".join(
        f'<div class="seg v-{k}" style="flex-grow:{counts[k]}" title="{NAMES[k]}: {counts[k]}">'
        f'{_ico(k)}{counts[k]}<span class="segname">&nbsp;{SHORT[k]}</span></div>'
        for k in VERDICTS if counts.get(k))
    bar_aria = ", ".join(f"{counts.get(k, 0)} {NAMES[k]}" for k in VERDICTS)
    chips = [f'<button type="button" class="f" data-f="ALL" aria-pressed="true">All <span class="n">{n}</span></button>']
    for k in VERDICTS:
        dis = "" if counts.get(k) else " disabled"
        chips.append(f'<button type="button" class="f v-{k}" data-f="{k}" aria-pressed="false"{dis}>'
                     f'{_ico(k)}{NAMES[k]} <span class="n">{counts.get(k, 0)}</span></button>')
    body = (
        f'<div id="vr" data-key="{key}">'
        f'<header class="head"><div class="hl"><h2>{esc(headline(counts, n))}</h2>'
        f'<div class="sub">{n} claim{"s" if n != 1 else ""} checked, split by {esc(eng.get("claims_by", "sentences"))}</div></div>'
        f'<div class="bar" role="img" aria-label="Verdicts: {bar_aria}">{segs}</div>'
        f'<div class="filters" role="group" aria-label="Show claims by verdict">{"".join(chips)}</div>'
        f'{_engine(eng)}</header>'
        f'<div class="grid"><section class="panel answer-panel" aria-labelledby="ah">'
        f'<h3 id="ah">Answer, sentence by sentence</h3>'
        f'<p class="hint">Hover or tab to a sentence to light up its claim; click or press Enter to jump to it.</p>'
        f'<div class="answer" id="answer">{_answer_html(report.answer, claims)}</div>'
        f'<p class="key">Markers show the claim number and its verdict symbol. Underlines: solid for '
        f'supported, double for refuted, dashed for unsettled.</p></section>'
        f'<section class="cards" aria-labelledby="ch"><div class="cards-h"><h3 id="ch">Claims</h3>'
        f'<span class="status" id="status" aria-live="polite">Showing all {n}</span></div>'
        f'<p class="olap"><mark>Highlighted words</mark> appear in both the claim and the passage. '
        f'This is simple word overlap, shown to help you read; it is not how the model decided.</p>'
        + "".join(_card(i, c, qhat, scifact) for i, c in enumerate(claims, 1))
        + '<p class="none" id="none" hidden>No claims with this verdict.</p></section></div></div>')
    doc = (f'<!doctype html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<style>{REPORT_CSS}</style><script>{REPORT_HEAD_JS.replace("__KEY__", key)}</script></head>'
           f'<body>{body}<script>{REPORT_JS}</script></body></html>')
    # first guess only: header + the taller column (cards on wide screens)
    guess = 300 + sum(330 + 90 * (len(c.evidence) > 1) + len(c.claim) // 2 for c in claims)
    return doc, max(420, guess)


# ------------------------------------------------------------------ state + callbacks
st.session_state.setdefault("n_sources", 1)
st.session_state.setdefault("source_mode", SRC_PASTE)
st.session_state.setdefault("answer", "")
# keep source fields alive while their widgets are hidden (SciFact mode)
for _k in [k for k in st.session_state if str(k).startswith("src_")]:
    st.session_state[_k] = st.session_state[_k]


def load_example():
    ex = EXAMPLES.get(st.session_state.get("example_pick") or "")
    st.session_state.example_pick = None          # pills act as buttons
    if not ex:
        return
    st.session_state.answer = ex["answer"]
    st.session_state.pop("report", None)
    if ex.get("scifact"):
        st.session_state.source_mode = SRC_SCIFACT
        return
    st.session_state.source_mode = SRC_PASTE
    st.session_state.n_sources = max(1, len(ex["sources"]))
    for i in range(MAX_SOURCES):
        name, text = ex["sources"][i] if i < len(ex["sources"]) else ("", "")
        st.session_state[f"src_name_{i}"] = name or f"source {i + 1}"
        st.session_state[f"src_text_{i}"] = text


def add_source():
    st.session_state.n_sources = min(MAX_SOURCES, st.session_state.n_sources + 1)


def remove_source():
    st.session_state.n_sources = max(1, st.session_state.n_sources - 1)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.html(
        '<div class="v-side-h">How it works</div><ol class="v-steps">'
        '<li data-n="1" style="--k:#e0313b;--k2:#e4553a"><b>Decompose.</b> Split the answer into claims: one per sentence, or an LLM '
        'rewrites sentences into single-fact claims.</li>'
        '<li data-n="2" style="--k:#e4553a;--k2:#e8762e"><b>Retrieve.</b> Find the passages most likely to settle each claim '
        '(BM25 keyword search combined with embeddings).</li>'
        '<li data-n="3" style="--k:#e8762e;--k2:#ec963c"><b>Window.</b> Cut the evidence into overlapping 2-sentence '
        'windows, so each passage the verifier reads stays short and specific.</li>'
        '<li data-n="4" style="--k:#ec963c;--k2:#f0b54a"><b>Verify.</b> A natural-language-inference model scores every '
        '(evidence, claim) pair as supported, refuted or not enough evidence.</li>'
        '<li data-n="5" style="--k:#f0b54a;--k2:#f0b54a"><b>Decide.</b> The passage with the strongest supporting or refuting score '
        'wins; conformal prediction turns its calibrated probabilities into a verdict, or '
        '<i>Uncertain</i>.</li></ol>'
        '<div class="v-side-h">Verdicts</div><div class="v-defs">'
        f'<div>{chip("SUPPORTED")}<br>The evidence states or implies the claim.</div>'
        f'<div>{chip("REFUTED")}<br>The evidence contradicts the claim.</div>'
        f'<div>{chip("NEI")}<br>The sources do not settle it either way. Not the same as false.</div>'
        f'<div>{chip("UNCERTAIN")}<br>At the calibrated 90% coverage level the verifier could not '
        'narrow it to one label; the card lists the labels it could not rule out.</div></div>'
        '<div class="v-side-h">One verifier</div><div class="v-defs">'
        '<div><b>Pasted sources</b> are cut into 2-sentence windows.</div>'
        '<div><b>SciFact corpus</b>: Verity first finds the 3 most relevant abstracts, then checks '
        'every 2-sentence window inside them.</div>'
        '<div>Both use the fine-tuned verifier with its own calibration. The SciFact design was chosen on '
        'held-out claims; pasted sources are cut the same way.</div></div>'
        '<div class="v-side-h">Limits</div><ul class="v-limits">'
        '<li>It judges only against the sources given. If they are wrong, Verity agrees with them.</li>'
        '<li>The 90% target is a long-run rate, calibrated on annotated evidence for science claims. '
        'It is not a promise about any single claim, and on retrieved evidence or other kinds of '
        'text the measured rate is lower (see the results).</li>'
        '<li>The fine-tuned verifier leans towards "not enough evidence" on true claims.</li>'
        '<li>Long sources are cut into two-sentence windows, so claims that need far-apart facts '
        'can be missed.</li>'
        '<li>Read the cited evidence before relying on a verdict.</li></ul>')

# ------------------------------------------------------------------ header
st.html('<header class="v-hero"><div class="v-head"><div class="v-core" aria-hidden="true"></div><div>'
        '<h1 class="v-title">VERITY</h1><div class="v-tag"><b>No strings.</b> <i>Only sources.</i></div>'
        '</div></div><div class="v-fracture" aria-hidden="true"></div></header><p class="v-lede">Check an AI-written answer claim by claim against the '
        'sources it should agree with. Every verdict comes with the passage that decided it.</p>')

# ------------------------------------------------------------------ inputs
col_a, col_s = st.columns([1.1, 1], gap="large")

with col_a:
    st.html('<div class="v-step"><span>01</span>The answer</div>')
    st.pills("Load an example", list(EXAMPLES), key="example_pick", on_change=load_example,
             help="Each of the first three examples has one claim the sources contradict and one "
                  "they do not cover. The SciFact example searches the science corpus.")
    st.text_area("AI answer to check", key="answer", height=208,
                 placeholder="Paste the answer an AI assistant gave you ...")
    has_key = groq_key_available()
    use_llm = st.toggle("Split into single-fact claims with an LLM", key="use_llm", disabled=not has_key,
                        help="Off: one claim per sentence, fully offline. On: the answer text is sent "
                             "to the Groq API to be split; verification itself stays local.")
    if not has_key:
        st.caption("LLM splitting is off: no GROQ_API_KEY was found. Sentence splitting runs offline.")
    elif use_llm:
        st.caption("The answer text will be sent to the Groq API to be split into claims.")

sources: list[tuple[str, str]] = []
with col_s:
    st.html('<div class="v-step"><span>02</span>The sources</div>')
    mode = st.radio("Check against", [SRC_PASTE, SRC_SCIFACT], key="source_mode", horizontal=True)
    if mode == SRC_PASTE:
        for i in range(int(st.session_state.n_sources)):
            st.session_state.setdefault(f"src_name_{i}", f"source {i + 1}")
            with st.container(border=True):
                st.text_input(f"Source {i + 1} name", key=f"src_name_{i}",
                              placeholder="File name or title")
                st.text_area(f"Source {i + 1} text", key=f"src_text_{i}", height=96,
                             label_visibility="collapsed",
                             placeholder="Paste the text this answer should agree with ...")
            text = st.session_state.get(f"src_text_{i}", "").strip()
            if text:
                sources.append((st.session_state[f"src_name_{i}"].strip() or f"source {i + 1}", text))
        b1, b2 = st.columns(2)
        b1.button("Add source", icon=":material/add:", on_click=add_source, width="stretch",
                  disabled=st.session_state.n_sources >= MAX_SOURCES)
        b2.button("Remove last", icon=":material/remove:", on_click=remove_source, width="stretch",
                  disabled=st.session_state.n_sources <= 1)
        files = st.file_uploader("Or upload .txt / .md files", type=["txt", "md"],
                                 accept_multiple_files=True)
        for f in files or []:
            text = f.getvalue().decode("utf-8", errors="replace").strip()
            if text:
                sources.append((f.name, text))
        # Verity names passages "<name>#<j>", so duplicate names would merge two sources
        seen: dict[str, int] = {}
        for i, (n, t) in enumerate(sources):
            seen[n] = seen.get(n, 0) + 1
            if seen[n] > 1:
                sources[i] = (f"{n} ({seen[n]})", t)
        st.caption("Cut into 2-sentence windows and checked by the fine-tuned verifier, with its own calibration."
                   if FINETUNED_DIR.exists() else "Cut into 2-sentence windows and checked by the zero-shot verifier.")
    else:
        st.html('<div class="v-panel"><div class="v-h" style="margin-top:0">SciFact science corpus</div>'
                '<p style="margin:0 0 .5rem">5,183 biomedical research abstracts. For each claim Verity '
                'finds the 3 most relevant abstracts, then the fine-tuned verifier checks every '
                '2-sentence window inside them.</p>'
                '<p class="v-muted" style="margin:0;font-size:.85rem">The first check in a session loads '
                'the corpus and its embeddings, which takes longer. A claim outside biomedicine will '
                'usually come back as not enough evidence: that is the corpus speaking, not a judgement '
                'that the claim is false.</p></div>')

st.write("")
run = st.button("Check answer", type="primary", icon=":material/fact_check:", width="stretch")
st.html('<hr class="v-merge" aria-hidden="true">')

# ------------------------------------------------------------------ run
out = st.container()
if run:
    answer = st.session_state.get("answer", "")
    if not answer.strip():
        out.warning("Paste an answer to check first, or load an example.", icon=":material/edit_note:")
    elif mode == SRC_PASTE and not sources:
        out.warning("Add the text of at least one source, or switch to the SciFact corpus.",
                    icon=":material/library_add:")
    else:
        scan = out.empty()
        msg = ("Loading the SciFact corpus and checking claims" if mode == SRC_SCIFACT
               else "Checking claims against your sources")
        scan.html(f'<div class="v-panel" role="status"><div class="v-scan-label">{msg}</div>'
                  '<div class="v-scan"></div></div>')
        try:
            t0 = time.perf_counter()
            base = scifact_verity() if mode == SRC_SCIFACT else paste_verity(tuple(sources))
            v = copy.copy(base)          # shallow copy: the cached object is shared by sessions
            v.decomposer = "llm" if use_llm else "sentences"
            report = v.check(answer)
            if mode == SRC_SCIFACT:
                scifact_titles()
            st.session_state.report = report
            st.session_state.engine = {
                "mode": "SciFact corpus (5,183 abstracts)" if mode == SRC_SCIFACT
                        else f"pasted sources ({len(sources)})",
                "scifact": mode == SRC_SCIFACT, "verifier": v.verifier, "temperature": v.temperature,
                "qhat": v.qhat, "select": v.select, "top_k": getattr(v, "top_k", None), "seconds": time.perf_counter() - t0,
                "claims_by": "LLM" if use_llm else "sentences"}
        except Exception as e:                       # shown to the user, not swallowed
            st.session_state.pop("report", None)
            out.error(f"The check could not run: {e}", icon=":material/error:")
        finally:
            scan.empty()

# ------------------------------------------------------------------ output
report = st.session_state.get("report")
with out:
    if report is None:
        if not run:
            st.html('<div class="v-step v-order"><span class="v-hex" aria-hidden="true"></span>'
                    '<span>03</span>The report</div>'
                    '<div class="v-panel v-empty"><div class="v-h" style="margin:0 0 .3rem">'
                    'Results appear here</div><p class="v-muted">Paste an answer and its sources, or '
                    'load an example, then press <b>Check answer</b>. Each claim comes back as one of:</p>'
                    f'<div class="v-legend">{"".join(chip(k) for k in VERDICTS)}</div></div>')
    elif not report.claims:
        st.info("No checkable sentences were found in the answer.", icon=":material/info:")
    else:
        eng = st.session_state.get("engine", {})
        st.html('<div class="v-step v-order"><span class="v-hex" aria-hidden="true"></span>'
                '<span>03</span>The report</div>')
        doc, guess = report_component(report, eng, theme_base())
        if hasattr(st, "iframe"):          # components.v1.html is deprecated in newer Streamlit
            st.iframe(doc, height=guess)
        else:
            components.html(doc, height=guess)
        c1, c2 = st.columns([1, 2], gap="medium", vertical_alignment="center")
        c1.download_button("Download report (JSON)", report.to_json(), file_name="verity_report.json",
                           mime="application/json", icon=":material/download:", width="stretch")
        c2.caption("Verdicts are judged only against the sources listed, by a model that can be wrong. "
                   "Read the decisive passage before relying on a verdict.")

# ------------------------------------------------------------------ under the hood
st.write("")
with st.expander("Under the hood: what was measured", icon=":material/science:"):
    headline_md = results_block()
    if headline_md:
        st.markdown(headline_md)
        st.caption("Read at run time from the results block of README.md, which is generated from results/.")
    else:
        st.caption("No results block found in README.md.")
    figs = [(t, ROOT / "figures" / f) for t, f in FIGURES if (ROOT / "figures" / f).exists()]
    if figs:
        for tab, (_, path) in zip(st.tabs([t for t, _ in figs]), figs):
            tab.image(str(path), width="stretch")
