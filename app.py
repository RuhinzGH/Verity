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

from verity.files import TYPES as FILE_TYPES, read_text
from verity.text import STOPWORDS, WORD, claim_parts, content_words, stem

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
REPORT_FONTS = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
                "&family=Inter+Tight:wght@600;700;800&family=JetBrains+Mono:wght@500;600&display=swap")
REPORT_CSS = r"""
/* --brand is the cyan of the page's beam (focus rings, the All chip, the engine line);
   --scan is the indigo the answer shows while the reveal decodes it. Neither marks a verdict. */
:root { color-scheme: dark;
  --bg:#05060b; --panel:#0c0f1b; --panel2:#121629; --line:#1f2539; --text:#e6eaf8; --muted:#9aa3bf;
  --brand:#22d3ee; --brand-t:#67e8f9; --scan:#6366f1; --scan-t:#a5b4fc; --ink:#06080f;
  --ok:#34d399; --ok-t:#6ee7b7; --bad:#fb7185; --bad-t:#fda4af; --nei:#94a3b8; --nei-t:#cbd5e1;
  --unc:#fbbf24; --unc-t:#fcd34d; --mark:rgba(34,211,238,.20); --mark-line:rgba(34,211,238,.7);
  --shadow:0 18px 44px -24px rgba(0,0,0,.9); }
[data-theme="light"] { color-scheme: light;
  --bg:#f6f7fb; --panel:#ffffff; --panel2:#f1f4fb; --line:#dde1ec; --text:#141a2e; --muted:#525b75;
  --brand:#0891b2; --brand-t:#0e7490; --scan:#4f46e5; --scan-t:#4338ca; --ink:#06080f;
  --ok:#34d399; --ok-t:#047857; --bad:#fb7185; --bad-t:#be123c; --nei:#94a3b8; --nei-t:#475569;
  --unc:#fbbf24; --unc-t:#b45309; --mark:rgba(8,145,178,.16); --mark-line:rgba(8,145,178,.7);
  --shadow:0 14px 34px -22px rgba(20,30,60,.35); }
.v-SUPPORTED { --c:var(--ok); --ct:var(--ok-t); } .v-REFUTED { --c:var(--bad); --ct:var(--bad-t); }
.v-NEI { --c:var(--nei); --ct:var(--nei-t); } .v-UNCERTAIN { --c:var(--unc); --ct:var(--unc-t); }
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); }
body { font: 16px/1.5 "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased; overflow: hidden; }
#vr { padding: 2px 2px 6px; }
h2, h3, p { margin: 0; }
button { font: inherit; color: inherit; }
:focus-visible { outline: 2px solid var(--brand); outline-offset: 2px; }
.ico { display: inline-flex; align-items: center; justify-content: center; flex: none; width: 1.1rem; height: 1.1rem;
  border-radius: 50%; font-size: .68rem; font-weight: 800; line-height: 1; background: var(--c); color: var(--ink); }
.chip { display: inline-flex; align-items: center; gap: .38rem; padding: .14rem .62rem .14rem .28rem; border-radius: 999px;
  font-size: .8rem; font-weight: 650; white-space: nowrap; color: var(--ct);
  background: color-mix(in srgb, var(--c) 13%, transparent); border: 1px solid color-mix(in srgb, var(--c) 50%, transparent); }

/* ---------- header */
.hl { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: .3rem 1rem; }
.hl h2 { font: 750 1.6rem/1.2 "Inter Tight", "Inter", sans-serif; letter-spacing: -.02em; }
.sub { font-size: .86rem; color: var(--muted); }
.bar { display: flex; gap: 3px; height: 2.2rem; margin: 1rem 0 .9rem; border-radius: 999px; overflow: hidden; }
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
.f[data-f="ALL"] { padding-left: .8rem; --c: var(--brand); --ct: var(--text); }
.f .n { font-variant-numeric: tabular-nums; font-weight: 750; padding: 0 .38rem; border-radius: 999px; min-width: 1.4rem; text-align: center;
  background: color-mix(in srgb, var(--text) 10%, transparent); }
.f:hover:not(:disabled) { background: color-mix(in srgb, var(--c) 20%, transparent); }
.f[aria-pressed="true"] { background: color-mix(in srgb, var(--c) 24%, transparent); border-color: var(--c);
  box-shadow: inset 0 0 0 1px var(--c); }
.f[aria-pressed="true"]::after { content: "\2713"; font-size: .72rem; margin-left: -.1rem; }
.f:disabled { cursor: default; opacity: .62; border-style: dashed; }
.engine { display: flex; flex-wrap: wrap; align-items: baseline; row-gap: .1rem; margin-top: .9rem; padding-top: .6rem;
  border-top: 1px solid color-mix(in srgb, var(--brand) 24%, transparent);
  font: .74rem/1.6 "JetBrains Mono", ui-monospace, Consolas, monospace; color: var(--muted); }
.engine b { color: var(--brand-t); font-weight: 600; letter-spacing: .12em; }
.engine span { white-space: nowrap; }
.engine i { font-style: normal; opacity: .6; padding: 0 .4rem; }

/* ---------- layout */
.grid { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); gap: 1.6rem; margin-top: 1.5rem; align-items: start; }
.panel { border: 1px solid var(--line); border-radius: 1.1rem; background: var(--panel);
  padding: 1rem 1.15rem 1.05rem; box-shadow: var(--shadow); }
h3 { font: 700 1.08rem/1.3 "Inter Tight", "Inter", sans-serif; letter-spacing: -.015em; }
.hint, .key { font-size: .8rem; line-height: 1.45; color: var(--muted); }
.hint { margin: .25rem 0 .75rem; }
.key { margin-top: .85rem; }

/* ---------- answer */
.answer-panel { will-change: transform; transition: transform .28s cubic-bezier(.2,.8,.2,1); }
.answer { font-size: 1.03rem; line-height: 2.05; overflow-wrap: anywhere; }
.s { cursor: pointer; border-radius: 5px; outline-offset: 1px; }
.s .t { padding: .12rem .16rem; border-radius: 4px; box-decoration-break: clone; -webkit-box-decoration-break: clone;
  background: color-mix(in srgb, var(--c) 14%, transparent); border-bottom: 2px solid color-mix(in srgb, var(--c) 80%, transparent);
  transition: background .18s, box-shadow .18s, color .25s; }
.s.v-REFUTED .t { border-bottom-style: double; border-bottom-width: 3px; }
.s.v-NEI .t, .s.v-UNCERTAIN .t { border-bottom-style: dashed; }
/* hover lifts the tint only: a ring drawn around each span would show a seam where the
   sentence's last word (kept on one line with its marker) begins */
.s:hover .t, .s.lit .t { background: color-mix(in srgb, var(--c) 32%, transparent); border-bottom-color: var(--c); }
.s:focus-visible { outline: 2px solid var(--brand); }
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
.card { position: relative; border: 1px solid var(--line); border-left: 3px solid var(--c); border-radius: 1rem;
  background: var(--panel); padding: .85rem 1rem .95rem; margin-bottom: .85rem; box-shadow: var(--shadow);
  transition: box-shadow .2s, border-color .2s, background .2s, transform .2s; scroll-margin: 1.2rem; }
.card[hidden] { display: none; }
.card.lit { border-color: color-mix(in srgb, var(--c) 70%, var(--line)); border-left-color: var(--c); background: var(--panel2);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--c) 55%, transparent), 0 0 26px -6px color-mix(in srgb, var(--c) 45%, transparent); }
.card:focus { outline: none; }
.card:focus-visible { outline: 2px solid var(--brand); outline-offset: 3px; }
.card.flash { animation: flash 1.1s ease-out; }
@keyframes flash { 0% { box-shadow: 0 0 0 3px var(--brand), 0 0 34px -4px var(--brand); } 100% { box-shadow: var(--shadow); } }
.s.flash .t { animation: sflash 1.1s ease-out; }
@keyframes sflash { 0% { box-shadow: 0 0 0 2px var(--brand); } 100% { box-shadow: 0 0 0 0 transparent; } }
.card-top { display: flex; align-items: center; justify-content: space-between; gap: .6rem; flex-wrap: wrap; }
.who { display: flex; align-items: center; gap: .55rem; }
.num { font: 600 .8rem "JetBrains Mono", ui-monospace, Consolas, monospace; color: var(--muted); }
.locate { display: inline-flex; align-items: center; gap: .3rem; cursor: pointer; font-size: .78rem; font-weight: 600;
  color: var(--muted); background: transparent; border: 1px solid var(--line); border-radius: 999px; padding: .12rem .6rem; }
.locate:hover { color: var(--text); border-color: color-mix(in srgb, var(--brand) 60%, var(--line)); }
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

/* ---------- reveal: the answer arrives as scrambled indigo noise and decodes into the verdict view */
.revealing .s:not(.settled) .t { color: var(--scan-t); background: color-mix(in srgb, var(--scan) 10%, transparent);
  border-bottom-color: color-mix(in srgb, var(--scan) 55%, transparent); border-bottom-style: solid;
  text-shadow: -1px 0 rgba(99,102,241,.55), 1px 0 rgba(34,211,238,.4); }
.revealing .s:not(.settled) .mark { opacity: 0; transform: scale(.6); }
.revealing .card { opacity: 0; transform: translateY(10px); border-left-color: var(--scan); }
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

# Runs in <head>, before the body is parsed: decide whether to play the reveal.
REPORT_HEAD_JS = r"""
(function () {
  var d = document.documentElement;
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

  /* ---- a new report: scroll to its start (once), so the result is seen without hunting for it */
  var scrolled = false; try { scrolled = store && store.getItem('verity-scrolled-' + KEY) === '1'; } catch (e) {}
  if (!scrolled) {
    try {
      store && store.setItem('verity-scrolled-' + KEY, '1');
      var fe = window.frameElement;
      if (fe) { fe.style.scrollMarginTop = '96px'; fe.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' }); }
    } catch (e) {}
  }

  /* ---- reveal: scrambled noise decoding, sentence by sentence, into the verdict view */
  if (doc.classList.contains('revealing')) {
    try { store && store.setItem('verity-seen-' + KEY, '1'); } catch (e) {}
    var answer = document.getElementById('answer'), noise = '#%&*+=<>/\\|{}[]?!01ABEFHKMNRSTVXZ';
    answer.setAttribute('aria-busy', 'true');
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
st.set_page_config(page_title="Verity · is this answer actually true?",
                   page_icon=":material/fact_check:", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
/* Theme "scrutiny light": a cold cyan-to-indigo beam passing over the text. The brand
   gradient is decoration only and never marks a verdict; verdicts keep their own colours
   (green / rose / slate / amber), each with a glyph and a word. Tints are mixed with
   `currentColor`, so the same rules hold on Streamlit's light theme. */
:root {
  --v-cyan: #22d3ee; --v-blue: #3b82f6; --v-indigo: #6366f1;
  --v-grad: linear-gradient(120deg, #67e8f9 0%, #22d3ee 24%, #3b82f6 62%, #818cf8 100%);
  /* darker stops, so white text stays above 4.5:1 along the whole button */
  --v-grad-btn: linear-gradient(120deg, #0e7490 0%, #0369a1 38%, #1d4ed8 72%, #4338ca 100%);
  --v-ok: #34d399; --v-bad: #fb7185; --v-nei: #94a3b8; --v-unc: #fbbf24;
  --v-ease: cubic-bezier(.16, 1, .3, 1);
}
.v-SUPPORTED { --c: var(--v-ok); } .v-REFUTED { --c: var(--v-bad); }
.v-NEI { --c: var(--v-nei); } .v-UNCERTAIN { --c: var(--v-unc); }
.v-muted { color: color-mix(in srgb, currentColor 66%, transparent); }

/* ---------- page */
/* Streamlit's header bar is transparent but full width, so it sat over the nav links:
   let the pointer through everywhere except its own buttons */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"], [data-testid="stHeader"] * { pointer-events: none; }
[data-testid="stHeader"] button, [data-testid="stHeader"] button *, [data-testid="stHeader"] a,
[data-testid="stHeader"] [role="button"] { pointer-events: auto; }
[data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1200px; }
[data-testid="stMain"] { background:
  radial-gradient(1100px 560px at 88% -8%, rgba(59,130,246,.15), transparent 62%),
  radial-gradient(900px 520px at -8% 6%, rgba(34,211,238,.11), transparent 62%),
  radial-gradient(800px 600px at 50% 105%, rgba(99,102,241,.08), transparent 60%); }
/* a faint measuring grid behind the hero, fading out at the edges */
.v-top { position: relative; }
.v-top > * { position: relative; z-index: 1; }
.v-top::before { content: ""; position: absolute; inset: -3rem 0 0; z-index: 0; pointer-events: none;
  background-image: linear-gradient(color-mix(in srgb, currentColor 5%, transparent) 1px, transparent 1px),
                    linear-gradient(90deg, color-mix(in srgb, currentColor 5%, transparent) 1px, transparent 1px);
  background-size: 56px 56px;
  -webkit-mask-image: radial-gradient(ellipse 70% 80% at 50% 0%, #000 20%, transparent 75%);
  mask-image: radial-gradient(ellipse 70% 80% at 50% 0%, #000 20%, transparent 75%); }

/* ---------- top bar */
.v-nav { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: .2rem 0 2.6rem; }
.v-brand { display: flex; align-items: center; gap: .6rem; font: 800 1.32rem/1 "Inter Tight", "Inter", sans-serif;
  letter-spacing: -.03em; color: inherit; text-decoration: none; }
.v-mark { position: relative; flex: none; width: 34px; height: 34px; border-radius: 11px; background: var(--v-grad);
  box-shadow: 0 0 18px rgba(34,211,238,.35); }
.v-mark::after { content: ""; position: absolute; left: 11px; top: 8px; width: 8px; height: 15px;
  border: solid #05060b; border-width: 0 3.5px 3.5px 0; border-radius: 1px; transform: rotate(45deg); }
.v-links { display: flex; align-items: center; gap: .4rem; flex-wrap: wrap; justify-content: flex-end; }
.v-links a { font-size: .88rem; font-weight: 550; padding: .45rem .85rem; border-radius: 999px; text-decoration: none;
  color: color-mix(in srgb, currentColor 75%, transparent); transition: color .2s, background .2s; }
.v-links a:hover { color: inherit; background: color-mix(in srgb, currentColor 7%, transparent); }
.v-links a.v-ghost { border: 1px solid color-mix(in srgb, currentColor 16%, transparent); color: inherit; }

/* ---------- hero */
.v-hero { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, .85fr); gap: 3rem; align-items: center;
  padding-bottom: 3.2rem; }
.v-pill { display: inline-flex; align-items: center; gap: .55rem; padding: .38rem .9rem .38rem .7rem; border-radius: 999px;
  font-size: .82rem; font-weight: 600; border: 1px solid color-mix(in srgb, currentColor 15%, transparent);
  background: color-mix(in srgb, currentColor 4%, transparent); }
.v-pill i { width: .5rem; height: .5rem; border-radius: 50%; background: var(--v-cyan);
  box-shadow: 0 0 0 4px rgba(34,211,238,.16), 0 0 12px var(--v-cyan); }
.v-h1 { font: 800 clamp(2.6rem, 5.6vw, 4.5rem)/1.02 "Inter Tight", "Inter", sans-serif; letter-spacing: -.032em;
  margin: 1.2rem 0 1.1rem; padding: 0; }
/* stops mixed with the text colour: paler on the dark theme, deeper (readable) on the light one */
.v-grad-text { background: linear-gradient(120deg, color-mix(in srgb, #22d3ee 78%, currentColor) 0%,
    color-mix(in srgb, #3b82f6 86%, currentColor) 62%, color-mix(in srgb, #818cf8 86%, currentColor) 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: var(--v-blue); padding-right: .04em; }
.v-lede { font-size: clamp(1.02rem, 1.3vw, 1.16rem); line-height: 1.62; max-width: 36rem; margin: 0;
  color: color-mix(in srgb, currentColor 74%, transparent); }
.v-stats { display: flex; flex-wrap: wrap; gap: 1.6rem 2.4rem; margin-top: 2rem; }
.v-stat b { display: block; font: 750 1.55rem/1 "Inter Tight", "Inter", sans-serif; letter-spacing: -.03em; }
.v-stat span { display: block; margin-top: .4rem; font-size: .8rem; color: color-mix(in srgb, currentColor 62%, transparent); }

/* the sample report beside the headline: what a checked answer looks like */
.v-demo { position: relative; border-radius: 1.4rem; padding: 1.3rem 1.3rem 1.1rem;
  background: color-mix(in srgb, currentColor 3.5%, transparent);
  border: 1px solid color-mix(in srgb, currentColor 10%, transparent);
  box-shadow: 0 0 0 1px rgba(34,211,238,.10), 0 30px 80px -30px rgba(59,130,246,.45); overflow: hidden; }
.v-demo::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: var(--v-grad); }
.v-demo::after { content: ""; position: absolute; left: 0; right: 0; top: -40%; height: 40%; pointer-events: none;
  background: linear-gradient(180deg, transparent, rgba(34,211,238,.10), transparent);
  animation: v-beam 5.5s var(--v-ease) infinite; }
@keyframes v-beam { 0% { top: -40%; } 60%, 100% { top: 110%; } }
.v-demo-h { display: flex; justify-content: space-between; align-items: center; font-size: .72rem; font-weight: 700;
  letter-spacing: .16em; text-transform: uppercase; color: color-mix(in srgb, currentColor 58%, transparent); margin-bottom: .9rem; }
.v-demo-h em { font-style: normal; letter-spacing: .04em; text-transform: none; font-weight: 600; }
.v-demo-row { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: .75rem;
  padding: .72rem .8rem; border-radius: .85rem; margin-bottom: .5rem;
  background: color-mix(in srgb, var(--c) 7%, transparent); border: 1px solid color-mix(in srgb, var(--c) 22%, transparent); }
.v-demo-row p { margin: 0; font-size: .92rem; line-height: 1.35; }
.v-demo-row small { font-size: .76rem; font-weight: 700; font-variant-numeric: tabular-nums;
  color: color-mix(in srgb, var(--c) 60%, currentColor); }
.v-demo-foot { font-size: .76rem; margin-top: .7rem; color: color-mix(in srgb, currentColor 55%, transparent); }

/* ---------- section labels */
.v-sec { display: flex; align-items: center; gap: .7rem; margin: .3rem 0 .9rem; }
.v-sec b { display: inline-flex; align-items: center; justify-content: center; width: 1.9rem; height: 1.9rem; border-radius: .6rem;
  font: 700 .78rem/1 "JetBrains Mono", ui-monospace, monospace; color: #04131a; background: var(--v-grad); }
.v-sec span { font: 700 1.12rem/1.2 "Inter Tight", "Inter", sans-serif; letter-spacing: -.015em; }
.v-sec small { margin-left: auto; font-size: .8rem; color: color-mix(in srgb, currentColor 58%, transparent); }
.v-eyebrow { font-size: .74rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase;
  color: color-mix(in srgb, currentColor 58%, transparent); margin: 0 0 .6rem; }
.v-h2 { font: 750 clamp(1.7rem, 3.2vw, 2.4rem)/1.1 "Inter Tight", "Inter", sans-serif; letter-spacing: -.026em; margin: 0 0 .5rem; padding: 0; }

/* ---------- input cards (keyed Streamlit containers) */
.st-key-card_answer, .st-key-card_sources {
  border-radius: 1.3rem; padding: 1.25rem 1.3rem 1.1rem;
  background: color-mix(in srgb, currentColor 3.2%, transparent);
  border: 1px solid color-mix(in srgb, currentColor 10%, transparent);
  box-shadow: 0 24px 60px -36px rgba(0,0,0,.7); }
[class*="st-key-src_box_"] { border-radius: 1rem; padding: .8rem .9rem .7rem;
  background: color-mix(in srgb, currentColor 2.5%, transparent);
  border: 1px solid color-mix(in srgb, currentColor 9%, transparent); }
[data-baseweb="textarea"], [data-baseweb="input"] { border-radius: .85rem !important; }
[data-baseweb="textarea"]:focus-within, [data-baseweb="input"]:focus-within {
  border-color: var(--v-cyan) !important; box-shadow: 0 0 0 3px rgba(34,211,238,.16); }
[data-testid="stFileUploaderDropzone"] { border-radius: .95rem; }

/* ---------- buttons */
[data-testid="stBaseButton-primary"] { background: var(--v-grad-btn) !important; color: #fff !important; border: 0 !important;
  min-height: 3.2rem; box-shadow: 0 0 0 1px rgba(99,102,241,.35), 0 18px 50px -18px rgba(37,99,235,.65);
  transition: transform .2s var(--v-ease), filter .2s; }
[data-testid="stBaseButton-primary"] p { color: #fff !important; font-weight: 700; font-size: 1.02rem; letter-spacing: .01em; }
[data-testid="stBaseButton-primary"]:hover { transform: translateY(-1px); filter: brightness(1.12); }
[data-testid="stBaseButton-primary"]:focus-visible { outline: 2px solid var(--v-cyan); outline-offset: 3px; }
[data-testid="stBaseButton-secondary"] { min-height: 3.2rem; }
[data-testid="stBaseButton-secondary"]:hover { border-color: color-mix(in srgb, var(--v-cyan) 60%, transparent); }
.st-key-src_actions [data-testid="stBaseButton-secondary"] { min-height: 2.5rem; }
[data-testid="stBaseButton-tertiary"] { color: color-mix(in srgb, currentColor 62%, transparent); }
[data-testid="stBaseButton-tertiary"]:hover { color: var(--v-cyan); }

/* ---------- verdict chips */
.v-chip { display: inline-flex; align-items: center; gap: .38rem; padding: .16rem .62rem .16rem .3rem;
  border-radius: 999px; font-size: .78rem; font-weight: 650; white-space: nowrap;
  color: color-mix(in srgb, var(--c) 58%, currentColor);
  background: color-mix(in srgb, var(--c) 12%, transparent); border: 1px solid color-mix(in srgb, var(--c) 42%, transparent); }
.v-ico { display: inline-flex; align-items: center; justify-content: center; width: 1.15rem; height: 1.15rem; flex: none;
  border-radius: 50%; font-size: .72rem; font-weight: 800; background: var(--c); color: #06080f; line-height: 1; }
.v-legend { display: flex; flex-wrap: wrap; gap: .45rem .55rem; }

/* ---------- empty report + checking */
.v-panel { border-radius: 1.2rem; padding: 1.2rem 1.3rem;
  background: color-mix(in srgb, currentColor 3%, transparent); border: 1px solid color-mix(in srgb, currentColor 10%, transparent); }
.v-panel h4 { font: 700 1.05rem/1.3 "Inter Tight", "Inter", sans-serif; margin: 0 0 .4rem; padding: 0; }
.v-panel p { margin: 0 0 .5rem; }
.v-empty { border-style: dashed; text-align: center; padding: 2rem 1.4rem; }
.v-empty .v-legend { justify-content: center; }
.v-empty p { margin: 0 auto .9rem; max-width: 34rem; }
.v-scan { position: relative; height: 3px; overflow: hidden; border-radius: 2px; margin: .7rem 0 .2rem;
  background: color-mix(in srgb, var(--v-cyan) 16%, transparent); }
.v-scan::after { content: ""; position: absolute; top: 0; left: -35%; width: 35%; height: 100%; background: var(--v-grad);
  box-shadow: 0 0 12px rgba(34,211,238,.7); animation: v-sweep 1.4s ease-in-out infinite; }
@keyframes v-sweep { from { left: -35%; } to { left: 100%; } }
.v-scan-label { font-size: .8rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
  color: color-mix(in srgb, var(--v-cyan) 65%, currentColor); }
.v-rule { height: 1px; border: 0; margin: 2.8rem 0 2.4rem;
  background: linear-gradient(90deg, transparent, color-mix(in srgb, currentColor 16%, transparent), transparent); }

/* ---------- how it works, verdicts, limits */
.v-steps { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .8rem; margin: 1.2rem 0 0; padding: 0; list-style: none; }
.v-steps li, .v-defs > div { border-radius: 1.1rem; padding: 1rem 1rem 1.05rem; margin: 0;
  background: color-mix(in srgb, currentColor 3%, transparent); border: 1px solid color-mix(in srgb, currentColor 9%, transparent);
  font-size: .87rem; line-height: 1.5; color: color-mix(in srgb, currentColor 78%, transparent); }
.v-steps li b, .v-defs b { display: block; font: 700 1rem/1.25 "Inter Tight", "Inter", sans-serif; margin: .7rem 0 .35rem;
  letter-spacing: -.01em; }
.v-steps .v-n { display: inline-flex; align-items: center; justify-content: center; width: 2rem; height: 2rem; border-radius: .65rem;
  font: 700 .8rem/1 "JetBrains Mono", ui-monospace, monospace; color: #04131a; background: var(--v-grad); }
.v-defs { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin-top: 1.2rem; }
.v-defs > div { border-top: 2px solid var(--c); }
.v-defs p { margin: .7rem 0 0; }
.v-limits { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .5rem 2rem; padding: 0; margin: 1rem 0 1.6rem; list-style: none; }
.v-limits li { position: relative; padding-left: 1.3rem; font-size: .9rem; line-height: 1.55;
  color: color-mix(in srgb, currentColor 76%, transparent); }
.v-limits li::before { content: ""; position: absolute; left: 0; top: .58rem; width: .45rem; height: .45rem; border-radius: 50%;
  background: var(--v-grad); }
.v-foot { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .6rem 1.2rem; margin-top: 3rem; padding-top: 1.3rem;
  border-top: 1px solid color-mix(in srgb, currentColor 10%, transparent);
  font-size: .82rem; color: color-mix(in srgb, currentColor 58%, transparent); }
.v-foot a { color: inherit; }
[data-testid="stExpander"] details { border-radius: 1.1rem; }

@media (max-width: 980px) {
  .v-hero { grid-template-columns: minmax(0, 1fr); gap: 2rem; }
  .v-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .v-defs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding-top: 1rem; }
  .v-nav { padding: 0 2.2rem 1.6rem 0; }       /* clear of Streamlit's menu button */
  .v-links a:not(.v-ghost) { display: none; }
  .v-steps, .v-defs, .v-limits { grid-template-columns: minmax(0, 1fr); }
  .v-stats { gap: 1.2rem 1.6rem; }
  .st-key-card_answer, .st-key-card_sources { padding: 1rem .95rem .9rem; }
}
@media (prefers-reduced-motion: reduce) { .v-demo::after, .v-scan::after { animation: none; } .v-scan::after { left: 32%; } }
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


@st.cache_data(show_spinner=False, max_entries=32)
def file_text(name: str, data: bytes) -> str:
    """The text of an uploaded file, read once per file (PDFs are slow to parse)."""
    return read_text(name, data)


FILE_NOTE = "Only the text is read: images and scanned pages (a picture of text) are not."


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
# components.html: no external scripts or URLs apart from the page's web fonts. The Python
# side builds all markup (so it reads correctly even before the script runs); the script only adds behaviour:
# linked highlighting, filters, the reveal, theme sync and sizing the iframe to its content.

def overlap_marked(passage: str, claim: str) -> tuple[str, int]:
    """The passage, escaped, with every word it shares with the claim wrapped in <mark>
    (normalised word overlap, stopwords removed). Adjacent shared words share one mark.
    Returns the HTML and the number of distinct shared words."""
    want = content_words(claim)
    out, pos, run_open, shared = [], 0, False, set()
    for m in WORD.finditer(passage):
        w = m.group()
        hit = w.lower() not in STOPWORDS and stem(w) in want
        gap = passage[pos:m.start()]
        if run_open and not (hit and gap.isspace()):
            out.append("</mark>")
            run_open = False
        out.append(esc(gap))
        if hit and not run_open:
            out.append("<mark>")
            run_open = True
        if hit:
            shared.add(stem(w))
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
             esc(f"{e['seconds']:.1f} s" + (f" on {e['threads']} CPU threads" if e.get("threads") else ""))]
    return ('<div class="engine"><b>ENGINE</b>'
            + "".join(f'<span><i aria-hidden="true">/</i>{x}</span>' for x in parts) + "</div>")


def _glued(text: str, marks: str) -> str:
    """The sentence text with its last word and markers kept on one line, so a marker never
    wraps away from the sentence it belongs to."""
    k = text.rstrip().rfind(" ") + 1
    head = f'<span class="t">{esc(text[:k])}</span>' if k else ""
    return f'{head}<span class="nw"><span class="t last">{esc(text[k:])}</span>{marks}</span>'


def _sentence_el(text: str, items: list[tuple[int, str]]) -> str:
    """One focusable piece of the answer, tinted by its worst verdict and followed by one
    marker (claim number + glyph) per claim it holds. Surrounding spaces stay outside."""
    body = text.strip()
    lead, trail = text[:len(text) - len(text.lstrip())], text[len(text.rstrip()):]
    worst = max((v for _, v in items), key=SEVERITY.get)
    ids = " ".join(str(i) for i, _ in items)
    who = ", ".join(f"claim {i} {NAMES[v]}" for i, v in items)
    marks = "".join(f'<span class="mark v-{v}" aria-hidden="true">{_ico(v)}{i}</span>' for i, v in items)
    return (f'{esc(lead)}<span class="s v-{worst}" role="button" tabindex="0" data-claims="{ids}" '
            f'aria-label="{esc(body)} ({who}). Press Enter to open the claim.">'
            f'{_glued(body, marks)}</span>{esc(trail)}')


def _answer_html(answer: str, claims) -> str:
    """The answer, sentence by sentence. A sentence the LLM split into several claims is
    coloured part by part when its parts can be told apart (verity.text.claim_parts);
    otherwise the whole sentence takes its worst verdict."""
    spans: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for i, c in enumerate(claims, 1):
        spans.setdefault((c.start, c.end), []).append((i, c.verdict))
    out, pos = [], 0
    for (a, e), items in sorted(spans.items()):
        if a < pos:                       # overlapping span: skip rather than duplicate text
            continue
        out.append(esc(answer[pos:a]))
        text = answer[a:e]
        parts = claim_parts(text, [claims[i - 1].claim for i, _ in items]) if len(items) > 1 else []
        if parts:
            placed = {j for _, _, j in parts}
            orphans = [it for j, it in enumerate(items) if j not in placed]   # marked at the end
            for k, (pa, pe, j) in enumerate(parts):
                own = [items[j]] + (orphans if k == len(parts) - 1 else [])
                out.append(_sentence_el(text[pa:pe], own))
        else:
            out.append(_sentence_el(text, items))
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
        lean = max(LABELS, key=lambda l: c.probs.get(l, 0))
        foot = (f'Leans <b>{NAMES[lean].lower()}</b> ({pct(c.probs.get(lean, 0))}), but at the 90% level '
                f'it could not rule out: <b>{esc(open_)}</b>.')
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
        f'supported, double for refuted, dashed for unsettled. A sentence split into several claims '
        f'is coloured part by part.</p></section>'
        f'<section class="cards" aria-labelledby="ch"><div class="cards-h"><h3 id="ch">Claims</h3>'
        f'<span class="status" id="status" aria-live="polite">Showing all {n}</span></div>'
        f'<p class="olap"><mark>Highlighted words</mark> appear in both the claim and the passage. '
        f'This is simple word overlap, shown to help you read; it is not how the model decided.</p>'
        + "".join(_card(i, c, qhat, scifact) for i, c in enumerate(claims, 1))
        + '<p class="none" id="none" hidden>No claims with this verdict.</p></section></div></div>')
    doc = (f'<!doctype html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<link rel="stylesheet" href="{REPORT_FONTS}"><style>{REPORT_CSS}</style><script>{REPORT_HEAD_JS.replace("__KEY__", key)}</script></head>'
           f'<body>{body}<script>{REPORT_JS}</script></body></html>')
    # first guess only: header + the taller column (cards on wide screens)
    guess = 300 + sum(330 + 90 * (len(c.evidence) > 1) + len(c.claim) // 2 for c in claims)
    return doc, max(420, guess)


# ------------------------------------------------------------------ state + callbacks
st.session_state.setdefault("n_sources", 1)
st.session_state.setdefault("source_mode", SRC_PASTE)
st.session_state.setdefault("answer", "")
st.session_state.setdefault("upload_gen", 0)       # bumped to empty the source file uploader
st.session_state.setdefault("answer_gen", 0)       # ... and the answer file uploader
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


def clear_answer():
    st.session_state.answer = ""
    st.session_state.pop("report", None)
    st.session_state.pop("answer_note", None)
    st.session_state.answer_gen += 1


def load_answer_file(key: str):
    """An uploaded answer replaces the text in the answer box (which can still be edited)."""
    f = st.session_state.get(key)
    st.session_state.pop("answer_note", None)
    if f is None:
        return
    text = file_text(f.name, f.getvalue())
    if text:
        st.session_state.answer = text
        st.session_state.pop("report", None)
    else:
        st.session_state.answer_note = f"No text found in {f.name}. {FILE_NOTE}"


def start_check():
    st.session_state.checking = True


def clear_source(i: int):
    st.session_state[f"src_name_{i}"] = f"source {i + 1}"
    st.session_state[f"src_text_{i}"] = ""


def clear_all():
    clear_answer()
    for i in range(MAX_SOURCES):
        clear_source(i)
    st.session_state.n_sources = 1
    st.session_state.upload_gen += 1


# ------------------------------------------------------------------ top bar + hero
LOGO = '<span class="v-mark" aria-hidden="true"></span>'     # drawn in CSS: st.html drops inline SVG
GITHUB, MODEL_HUB = "https://github.com/RuhinzGH/Verity", "https://huggingface.co/RuhinZ/verity-deberta-scifact"


def demo_row(v: str, text: str, p: str) -> str:
    return (f'<div class="v-demo-row v-{v}"><span class="v-ico" aria-hidden="true">{GLYPH[v]}</span>'
            f'<p>{text}</p><small>{SHORT[v]} &middot; {p}</small></div>')


st.html(
    f'<div class="v-top"><nav class="v-nav"><a class="v-brand" href="#check" aria-label="Verity">{LOGO}Verity</a>'
    '<div class="v-links"><a href="#how">How it works</a><a href="#limits">Limits</a>'
    f'<a href="{MODEL_HUB}" target="_blank" rel="noopener">Model</a>'
    f'<a class="v-ghost" href="{GITHUB}" target="_blank" rel="noopener">GitHub</a></div></nav>'
    '<section class="v-hero"><div>'
    '<span class="v-pill"><i aria-hidden="true"></i>No strings. Only sources.</span>'
    '<h1 class="v-h1">Is this answer<br><span class="v-grad-text">actually true?</span></h1>'
    '<p class="v-lede">Verity checks an AI-written answer claim by claim against the sources it should agree '
    'with, and shows the exact passage behind every verdict. When the evidence is not strong enough, '
    'it says so instead of guessing.</p>'
    '<div class="v-stats"><div class="v-stat"><b>0.786</b><span>verifier macro-F1, SciFact</span></div>'
    '<div class="v-stat"><b>5,183</b><span>science abstracts to search</span></div>'
    '<div class="v-stat"><b>90%</b><span>calibrated uncertainty target</span></div></div></div>'
    '<div class="v-demo" aria-label="Example of a checked answer"><div class="v-demo-h"><span>Sample report</span>'
    '<em>real verdicts from the examples</em></div>'
    + demo_row("SUPPORTED", "Jupiter is the largest planet in the Solar System.", "96%")
    + demo_row("REFUTED", "Mars is the third planet from the Sun.", "93%")
    + demo_row("NEI", "Saturn&rsquo;s rings are made mostly of water ice.", "96%")
    + demo_row("UNCERTAIN", "Vitamin D prevents all fractures in older adults.", "2 labels")
    + '<p class="v-demo-foot">Press <b>Planets</b> or <b>Smoking (SciFact)</b> below to run these yourself.</p></div></section></div>')

# ------------------------------------------------------------------ inputs
st.html('<div id="check"></div>')
col_a, col_s = st.columns([1.1, 1], gap="large")

with col_a, st.container(key="card_answer"):
    h, c = st.columns([4, 1], vertical_alignment="center")
    h.html('<div class="v-sec"><b>01</b><span>The answer</span></div>')
    c.button("Clear", key="clear_answer", type="tertiary", icon=":material/backspace:", on_click=clear_answer,
             help="Empty the answer box.")
    st.pills("Load an example", list(EXAMPLES), key="example_pick", on_change=load_example,
             help="Each of the first three examples has one claim the sources contradict and one "
                  "they do not cover. The SciFact example searches the science corpus.")
    st.text_area("AI answer to check", key="answer", height=208,
                 placeholder="Paste the answer an AI assistant gave you ...")
    _ak = f"answer_file_{st.session_state.answer_gen}"
    st.file_uploader("Or upload the answer: .txt, .md, .pdf or .docx", type=FILE_TYPES, key=_ak,
                     on_change=load_answer_file, args=(_ak,), help=FILE_NOTE)
    if st.session_state.get("answer_note"):
        st.warning(st.session_state.answer_note, icon=":material/description:")
    has_key = groq_key_available()
    use_llm = st.toggle("Split into single-fact claims with an LLM", key="use_llm", disabled=not has_key,
                        help="Off: one claim per sentence, fully offline. On: the answer text is sent "
                             "to the Groq API to be split; verification itself stays local.")
    if not has_key:
        st.caption("LLM splitting is off: no GROQ_API_KEY was found. Sentence splitting runs offline.")
    elif use_llm:
        st.caption("The answer text will be sent to the Groq API to be split into claims.")

sources: list[tuple[str, str]] = []
with col_s, st.container(key="card_sources"):
    st.html('<div class="v-sec"><b>02</b><span>The sources</span></div>')
    mode = st.radio("Check against", [SRC_PASTE, SRC_SCIFACT], key="source_mode", horizontal=True)
    if mode == SRC_PASTE:
        for i in range(int(st.session_state.n_sources)):
            st.session_state.setdefault(f"src_name_{i}", f"source {i + 1}")
            with st.container(key=f"src_box_{i}"):
                n, c = st.columns([4, 1], vertical_alignment="bottom")
                n.text_input(f"Source {i + 1} name", key=f"src_name_{i}", placeholder="File name or title")
                c.button("Clear", key=f"clear_src_{i}", type="tertiary", icon=":material/backspace:",
                         on_click=clear_source, args=(i,), help=f"Empty source {i + 1}.")
                st.text_area(f"Source {i + 1} text", key=f"src_text_{i}", height=96,
                             label_visibility="collapsed",
                             placeholder="Paste the text this answer should agree with ...")
            text = st.session_state.get(f"src_text_{i}", "").strip()
            if text:
                sources.append((st.session_state[f"src_name_{i}"].strip() or f"source {i + 1}", text))
        with st.container(key="src_actions"):
            b1, b2 = st.columns(2)
            b1.button("Add source", icon=":material/add:", on_click=add_source, width="stretch",
                      disabled=st.session_state.n_sources >= MAX_SOURCES)
            b2.button("Remove last", icon=":material/remove:", on_click=remove_source, width="stretch",
                      disabled=st.session_state.n_sources <= 1)
        files = st.file_uploader("Or upload sources: .txt, .md, .pdf or .docx", type=FILE_TYPES,
                                 accept_multiple_files=True, key=f"uploads_{st.session_state.upload_gen}",
                                 help=FILE_NOTE)
        empty = []
        for f in files or []:
            text = file_text(f.name, f.getvalue())
            if text:
                sources.append((f.name, text))
            else:
                empty.append(f.name)
        if empty:
            st.warning(f"No text found in {', '.join(empty)}. {FILE_NOTE}", icon=":material/description:")
        # Verity names passages "<name>#<j>", so duplicate names would merge two sources
        seen: dict[str, int] = {}
        for i, (n, t) in enumerate(sources):
            seen[n] = seen.get(n, 0) + 1
            if seen[n] > 1:
                sources[i] = (f"{n} ({seen[n]})", t)
        st.caption("Cut into 2-sentence windows and checked by the fine-tuned verifier, with its own calibration."
                   if FINETUNED_DIR.exists() else "Cut into 2-sentence windows and checked by the zero-shot verifier.")
    else:
        st.html('<div class="v-panel"><h4>SciFact science corpus</h4>'
                '<p>5,183 biomedical research abstracts. For each claim Verity finds the 3 most relevant '
                'abstracts, then the fine-tuned verifier checks every 2-sentence window inside them.</p>'
                '<p class="v-muted" style="margin:0;font-size:.85rem">The first check in a session loads '
                'the corpus and its embeddings, which takes longer. A claim outside biomedicine will '
                'usually come back as not enough evidence: that is the corpus speaking, not a judgement '
                'that the claim is false.</p></div>')

st.write("")
r1, r2 = st.columns([4, 1], gap="small")
checking = st.session_state.get("checking", False)
r1.button("Checking ..." if checking else "Check answer", key="run_btn", type="primary", width="stretch",
          icon=":material/hourglass_top:" if checking else ":material/fact_check:", disabled=checking,
          on_click=start_check)
r2.button("Clear all", icon=":material/restart_alt:", on_click=clear_all, width="stretch",
          help="Empty the answer, every source and the uploaded files.")
st.html('<hr class="v-rule" aria-hidden="true">')

# ------------------------------------------------------------------ run
# Pressing the button only sets `checking`; this run shows "Checking ..." on a disabled
# button while the check works, then reruns so the button and the report are redrawn.
out = st.container()
note = st.session_state.pop("flash", None)
if note:
    getattr(out, note[0])(note[1], icon=note[2])
if checking:
    st.session_state.checking = False
    answer = st.session_state.get("answer", "")
    if not answer.strip():
        st.session_state.flash = ("warning", "Paste an answer to check first, or load an example.",
                                  ":material/edit_note:")
    elif mode == SRC_PASTE and not sources:
        st.session_state.flash = ("warning", "Add the text of at least one source, or switch to the SciFact corpus.",
                                  ":material/library_add:")
    else:
        scan = out.empty()
        msg = ("Loading the SciFact corpus and checking claims" if mode == SRC_SCIFACT
               else "Checking claims against your sources")
        scan.html(f'<div class="v-panel" role="status"><div class="v-scan-label">{msg}</div>'
                  '<div class="v-scan"></div></div>')
        try:
            import torch
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
                "threads": torch.get_num_threads(),
                "claims_by": "LLM" if use_llm else "sentences"}
        except Exception as e:                       # shown to the user, not swallowed
            st.session_state.pop("report", None)
            st.session_state.flash = ("error", f"The check could not run: {e}", ":material/error:")
        scan.empty()
    st.rerun()

# ------------------------------------------------------------------ output
report = st.session_state.get("report")
with out:
    if report is None:
        if not note:
            st.html('<div class="v-sec"><b>03</b><span>The report</span></div>'
                    '<div class="v-panel v-empty"><h4>Results appear here</h4><p class="v-muted">Paste an answer '
                    'and its sources, or load an example, then press <b>Check answer</b>. Each claim comes '
                    'back as one of:</p>'
                    f'<div class="v-legend">{"".join(chip(k) for k in VERDICTS)}</div></div>')
    elif not report.claims:
        st.info("No checkable sentences were found in the answer.", icon=":material/info:")
    else:
        eng = st.session_state.get("engine", {})
        st.html('<div class="v-sec"><b>03</b><span>The report</span></div>')
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

# ------------------------------------------------------------------ how it works, verdicts, limits
STEPS = [
    ("Decompose", "Split the answer into claims: one per sentence, or an LLM rewrites sentences into "
                  "single-fact claims."),
    ("Retrieve", "Find the passages most likely to settle each claim: BM25 keyword search combined "
                 "with embeddings."),
    ("Window", "Cut the evidence into overlapping 2-sentence windows, so each passage the verifier "
               "reads stays short and specific."),
    ("Verify", "A fine-tuned DeBERTa-v3 natural-language-inference model scores every (evidence, "
               "claim) pair."),
    ("Decide", "The strongest supporting or refuting passage wins; conformal prediction turns its "
               "calibrated probabilities into a verdict, or <i>Uncertain</i>."),
]
DEFS = {"SUPPORTED": "The evidence states or implies the claim.",
        "REFUTED": "The evidence contradicts the claim.",
        "NEI": "The sources do not settle it either way. Not the same as false.",
        "UNCERTAIN": "At the calibrated 90% level the verifier could not narrow it to one label; "
                     "the card lists the labels it could not rule out."}
LIMITS = [
    "It judges only against the sources given. If they are wrong, Verity agrees with them.",
    "The 90% target is a long-run rate, calibrated on annotated evidence for science claims. It is "
    "not a promise about any single claim, and on retrieved evidence or other kinds of text the "
    "measured rate is lower (see the results).",
    "The fine-tuned verifier leans towards \"not enough evidence\" on true claims.",
    "Long sources are cut into two-sentence windows, so claims that need far-apart facts can be missed.",
    "Pasted sources and the SciFact corpus use the same verifier and calibration; the SciFact design "
    "was chosen on held-out claims.",
    "Read the cited evidence before relying on a verdict.",
]
st.html('<hr class="v-rule" aria-hidden="true"><section id="how"><p class="v-eyebrow">How it works</p>'
        '<h2 class="v-h2">From one answer to cited verdicts, in five steps</h2><ol class="v-steps">'
        + "".join(f'<li><span class="v-n">{k}</span><b>{t}</b>{d}</li>' for k, (t, d) in enumerate(STEPS, 1))
        + '</ol><div class="v-defs">'
        + "".join(f'<div class="v-{k}">{chip(k)}<p>{d}</p></div>' for k, d in DEFS.items())
        + '</div></section><hr class="v-rule" aria-hidden="true"><section id="limits">'
        '<p class="v-eyebrow">Limits</p><h2 class="v-h2">What Verity cannot promise</h2><ul class="v-limits">'
        + "".join(f"<li>{x}</li>" for x in LIMITS) + "</ul></section>")

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

st.html(f'<footer class="v-foot"><span>Verity &middot; Ruhin Zaidi &middot; 23215149 &middot; BCA (Honours) '
        f'Generative AI mini project</span><span><a href="{GITHUB}" target="_blank" rel="noopener">Source on '
        f'GitHub</a> &middot; <a href="{MODEL_HUB}" target="_blank" rel="noopener">Verifier on Hugging Face</a>'
        f'</span></footer>')
