"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           HYBRID SENTIMENT ANALYZER  ·  PRO EDITION                        ║
║  Models   : VADER · RoBERTa · DistilBERT (3-model ensemble)                ║
║  Features : Real-time voice · Batch CSV · Aspect-Based SA · Emotion wheel  ║
║             Session export (JSON/CSV/PDF) · Live chart · SQLite history     ║
║             Confidence calibration · Negation detection · Emoji/slang norm  ║
║  GUI      : CustomTkinter — Arctic Steel dark theme                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import csv
import io
import json
import os
import queue
import re
import sqlite3
import threading
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import StringVar, messagebox, filedialog

# ─────────────────────────────────────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────────────────────────────────────
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import nltk
import torch
import numpy as np
import speech_recognition as sr
from deep_translator import GoogleTranslator
from deep_translator.constants import GOOGLE_LANGUAGES_TO_CODES
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# NLTK one-time downloads
# ─────────────────────────────────────────────────────────────────────────────
for _pkg in ("vader_lexicon", "punkt", "stopwords", "averaged_perceptron_tagger"):
    nltk.download(_pkg, quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ROBERTA_MODEL    = "cardiffnlp/twitter-roberta-base-sentiment"
DISTILBERT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL    = "j-hartmann/emotion-english-distilroberta-base"
LABELS           = ["Negative", "Neutral", "Positive"]

DB_PATH  = Path.home() / ".sentiment_pro" / "history.db"
LOG_PATH = Path.home() / ".sentiment_pro" / "session.log"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Emoji → sentiment-neutral text normalisation map
EMOJI_MAP = {
    "😊": "happy", "😢": "sad", "😠": "angry", "❤️": "love",
    "👍": "good",  "👎": "bad", "🔥": "fire", "💔": "heartbreak",
    "😂": "funny", "😡": "furious", "😍": "amazing", "🙄": "sarcastic",
    "😭": "crying", "🥰": "loving", "😤": "frustrated",
}

# Internet-slang normalisation
SLANG_MAP = {
    r"\blol\b": "laughing", r"\bomg\b": "oh my god", r"\bwtf\b": "what the heck",
    r"\bngl\b": "not gonna lie", r"\bimo\b": "in my opinion",
    r"\bidk\b": "i don't know", r"\bbtw\b": "by the way",
    r"\bngl\b": "not gonna lie", r"\bsmh\b": "shaking my head",
    r"\bgoat\b": "greatest of all time", r"\blit\b": "amazing",
}

# Arctic Steel colour palette
C = {
    "bg":       "#07111a",
    "surface":  "#0c1f30",
    "tile":     "#102840",
    "accent":   "#7dd3fc",
    "accent2":  "#38bdf8",
    "low":      "#4ade80",
    "medium":   "#fbbf24",
    "high":     "#fb7185",
    "text":     "#e0f2fe",
    "muted":    "#3d6d8a",
    "border":   "#1a3d5c",
    "purple":   "#a78bfa",
    "teal":     "#2dd4bf",
}

SC = {   # (fg, bg) per sentiment
    "Positive": ("#4ade80", "#052e16"),
    "Neutral":  ("#fbbf24", "#2d1f00"),
    "Negative": ("#fb7185", "#2d0a10"),
}

EMOTION_COLORS = {
    "joy":      "#fbbf24",
    "sadness":  "#60a5fa",
    "anger":    "#fb7185",
    "fear":     "#a78bfa",
    "disgust":  "#34d399",
    "surprise": "#f97316",
    "neutral":  "#94a3b8",
}

# ─────────────────────────────────────────────────────────────────────────────
# Database layer
# ─────────────────────────────────────────────────────────────────────────────
class HistoryDB:
    """Persistent SQLite store for analysis results."""

    def __init__(self, path: Path = DB_PATH):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT NOT NULL,
                    original    TEXT NOT NULL,
                    normalized  TEXT,
                    final_label TEXT NOT NULL,
                    confidence  REAL,
                    vader_compound REAL,
                    roberta_neg REAL, roberta_neu REAL, roberta_pos REAL,
                    distilbert_label TEXT, distilbert_score REAL,
                    emotions    TEXT,
                    aspects     TEXT,
                    agreed      INTEGER,
                    source      TEXT DEFAULT 'typed',
                    lang_detected TEXT
                )
            """)

    def insert(self, row: dict):
        with self._lock:
            with self.conn:
                self.conn.execute("""
                    INSERT INTO analyses
                    (ts, original, normalized, final_label, confidence,
                     vader_compound, roberta_neg, roberta_neu, roberta_pos,
                     distilbert_label, distilbert_score,
                     emotions, aspects, agreed, source, lang_detected)
                    VALUES
                    (:ts, :original, :normalized, :final_label, :confidence,
                     :vader_compound, :roberta_neg, :roberta_neu, :roberta_pos,
                     :distilbert_label, :distilbert_score,
                     :emotions, :aspects, :agreed, :source, :lang_detected)
                """, row)

    def fetch_all(self, limit: int = 200) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def stats(self) -> dict:
        cur = self.conn.execute("""
            SELECT final_label, COUNT(*) as cnt,
                   AVG(confidence) as avg_conf
            FROM analyses GROUP BY final_label
        """)
        return {r[0]: {"count": r[1], "avg_conf": r[2]} for r in cur.fetchall()}

    def clear(self):
        with self._lock:
            with self.conn:
                self.conn.execute("DELETE FROM analyses")

    def export_csv(self, path: str):
        rows = self.fetch_all(limit=10_000)
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    def export_json(self, path: str):
        rows = self.fetch_all(limit=10_000)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessor
# ─────────────────────────────────────────────────────────────────────────────
class TextPreprocessor:
    """Normalises emojis, slang, URLs, handles negation boosting."""

    @staticmethod
    def normalise(text: str) -> str:
        # Emoji → word
        for emoji, word in EMOJI_MAP.items():
            text = text.replace(emoji, f" {word} ")
        # Slang
        for pattern, replacement in SLANG_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Mentions / hashtags (keep word)
        text = re.sub(r"[@#](\w+)", r"\1", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def detect_negation(text: str) -> bool:
        """Returns True if strong negation is present."""
        negs = re.compile(
            r"\b(not|no|never|neither|nor|don't|doesn't|didn't|can't|couldn't"
            r"|won't|wouldn't|isn't|aren't|wasn't|weren't|hardly|barely|scarcely)\b",
            re.IGNORECASE,
        )
        return bool(negs.search(text))

    @staticmethod
    def extract_aspects(text: str) -> list[str]:
        """
        Lightweight noun-phrase aspect extraction via POS tagging.
        Returns up to 5 noun phrases as candidate aspects.
        """
        try:
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            aspects = [
                word for word, tag in tagged
                if tag.startswith("NN") and len(word) > 3
            ]
            return list(dict.fromkeys(aspects))[:5]   # dedupe, limit
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# NLP Engine — 3-model ensemble
# ─────────────────────────────────────────────────────────────────────────────
class SentimentEngine:
    """
    Three-model ensemble:
      1. VADER       — rule-based, fast
      2. RoBERTa     — Twitter-trained transformer
      3. DistilBERT  — SST-2 fine-tuned transformer

    Weighted majority vote → final label.
    Also runs emotion classification (7 classes).
    """

    WEIGHTS = {"vader": 0.20, "roberta": 0.50, "distilbert": 0.30}

    def __init__(self, status_cb=None):
        # status_cb is a plain Python callable — safe from any thread.
        # The App routes it through a queue so Tkinter is never touched here.
        self._cb   = status_cb or (lambda m: None)
        self.ready = False
        self.pp    = TextPreprocessor()
        # Small delay so App.__init__ finishes before the thread fires
        threading.Timer(0.1, lambda: threading.Thread(
            target=self._load, daemon=True
        ).start()).start()

    # ── Loader ───────────────────────────────────────────────────────────────
    def _load(self):
        try:
            self._cb("Loading VADER …")
            self.sia = SentimentIntensityAnalyzer()

            self._cb("Loading RoBERTa tokenizer …")
            self.rob_tok   = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
            self._cb("Loading RoBERTa model …")
            self.rob_model = AutoModelForSequenceClassification.from_pretrained(
                ROBERTA_MODEL
            )
            self.rob_model.eval()

            self._cb("Loading DistilBERT …")
            self.dbert = pipeline(
                "text-classification",
                model=DISTILBERT_MODEL,
                truncation=True,
                max_length=512,
            )

            self._cb("Loading Emotion classifier …")
            self.emotion_clf = pipeline(
                "text-classification",
                model=EMOTION_MODEL,
                top_k=None,
                truncation=True,
                max_length=512,
            )

            self.ready = True
            self._cb("All models ready ✓")
        except Exception as e:
            self._cb(f"Load error: {e}")
            traceback.print_exc()

    # ── VADER ─────────────────────────────────────────────────────────────────
    def _run_vader(self, text: str) -> dict:
        scores   = self.sia.polarity_scores(text)
        compound = scores["compound"]
        label    = (
            "Positive" if compound >  0.05 else
            "Negative" if compound < -0.05 else
            "Neutral"
        )
        # Map compound to pseudo-probability for weighting
        prob_pos = (compound + 1) / 2
        return {
            "label":    label,
            "compound": compound,
            "scores":   scores,
            "probs":    {
                "Positive": prob_pos,
                "Neutral":  1 - abs(compound),
                "Negative": 1 - prob_pos,
            },
        }

    # ── RoBERTa ───────────────────────────────────────────────────────────────
    def _run_roberta(self, text: str) -> dict:
        inputs = self.rob_tok(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.rob_model(**inputs).logits[0]
        probs = softmax(logits.numpy())
        idx   = int(probs.argmax())
        return {
            "label":      LABELS[idx],
            "confidence": float(probs[idx]),
            "probs":      {LABELS[i]: float(probs[i]) for i in range(3)},
        }

    # ── DistilBERT ────────────────────────────────────────────────────────────
    def _run_distilbert(self, text: str) -> dict:
        out   = self.dbert(text)[0]
        raw   = out["label"].upper()         # "POSITIVE" | "NEGATIVE"
        label = "Positive" if raw == "POSITIVE" else "Negative"
        score = float(out["score"])
        return {
            "label":      label,
            "score":      score,
            "probs": {
                "Positive": score if label == "Positive" else 1 - score,
                "Neutral":  0.0,
                "Negative": score if label == "Negative" else 1 - score,
            },
        }

    # ── Emotion ───────────────────────────────────────────────────────────────
    def _run_emotions(self, text: str) -> dict:
        results = self.emotion_clf(text)[0]
        return {r["label"]: round(r["score"], 4) for r in results}

    # ── Ensemble ──────────────────────────────────────────────────────────────
    def _ensemble(self, v: dict, r: dict, d: dict) -> tuple[str, float]:
        """Weighted average of probabilities → final label + confidence."""
        w = self.WEIGHTS
        labels = ["Positive", "Neutral", "Negative"]
        avg = {}
        for lbl in labels:
            avg[lbl] = (
                w["vader"]      * v["probs"].get(lbl, 0) +
                w["roberta"]    * r["probs"].get(lbl, 0) +
                w["distilbert"] * d["probs"].get(lbl, 0)
            )
        final = max(avg, key=avg.get)
        confidence = avg[final]
        return final, round(confidence, 4), avg

    # ── Public API ────────────────────────────────────────────────────────────
    def analyze(self, text: str, source: str = "typed") -> dict:
        if not self.ready:
            raise RuntimeError("Models still loading — please wait.")

        normalized = self.pp.normalise(text)
        negation   = self.pp.detect_negation(text)
        aspects    = self.pp.extract_aspects(text)

        vader     = self._run_vader(normalized)
        roberta   = self._run_roberta(normalized)
        distilbert = self._run_distilbert(normalized)
        emotions  = self._run_emotions(normalized)

        final, conf, avg_probs = self._ensemble(vader, roberta, distilbert)

        # Negation nudge: if negation detected and model is weakly positive → dampen
        if negation and final == "Positive" and conf < 0.65:
            final = "Neutral"

        agreement = (
            vader["label"] == roberta["label"] == distilbert["label"]
        )

        return {
            "original":         text,
            "normalized":       normalized,
            "final":            final,
            "confidence":       conf,
            "ensemble_probs":   avg_probs,
            "agreed":           agreement,
            "negation":         negation,
            "aspects":          aspects,
            "vader":            vader,
            "roberta":          roberta,
            "distilbert":       distilbert,
            "emotions":         emotions,
            "source":           source,
            "ts":               datetime.now().isoformat(timespec="seconds"),
        }

    def translate(self, text: str, dest_lang: str) -> str:
        return GoogleTranslator(source="auto", target=dest_lang).translate(text)

    def batch_analyze(
        self,
        texts: list[str],
        progress_cb=None,
        source: str = "batch",
    ) -> list[dict]:
        results = []
        total = len(texts)
        for i, t in enumerate(texts, 1):
            try:
                results.append(self.analyze(t, source=source))
            except Exception as e:
                results.append({"original": t, "error": str(e)})
            if progress_cb:
                progress_cb(i / total)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Voice Capture
# ─────────────────────────────────────────────────────────────────────────────
class VoiceCapture:
    def __init__(self, timeout: int = 7):
        self.rec     = sr.Recognizer()
        self.timeout = timeout

    def capture(self) -> str:
        with sr.Microphone() as src:
            self.rec.adjust_for_ambient_noise(src, duration=0.6)
            audio = self.rec.listen(src, timeout=self.timeout, phrase_time_limit=30)
        return self.rec.recognize_google(audio)


# ─────────────────────────────────────────────────────────────────────────────
# Chart helper (matplotlib embedded in CTk)
# ─────────────────────────────────────────────────────────────────────────────
class EmbeddedChart:
    """Reusable dark-themed matplotlib canvas."""

    BG = "#07111a"

    def __init__(self, parent, figsize=(5, 2.2)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.patch.set_facecolor(self.BG)
        self.ax.set_facecolor(self.BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear(self):
        self.ax.cla()
        self.ax.set_facecolor(self.BG)

    def draw(self):
        self.canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────
class App(ctk.CTk):

    def __init__(self):
        super().__init__()

        # Window
        self.title("Hybrid Sentiment Analyzer  ·  Pro Edition")
        self.geometry("920x980")
        self.minsize(800, 860)
        self.configure(fg_color=C["bg"])

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── Thread-safe status queue ──────────────────────────────────────────
        # Background threads put status strings here; a main-thread poll drains
        # it via self.after() — this is the ONLY safe pattern for Tkinter.
        self._status_queue: queue.Queue = queue.Queue()

        # Core objects — engine gets a plain lambda (no Tkinter calls in it)
        self.engine     = SentimentEngine(
            status_cb=lambda msg: self._status_queue.put(msg)
        )
        self.voice_cap  = VoiceCapture()
        self.db         = HistoryDB()
        self.is_listening = False
        self._lang_map  = {
            v.title(): k for k, v in GOOGLE_LANGUAGES_TO_CODES.items()
        }
        self._session_results: list[dict] = []

        self._build_ui()
        self._refresh_stats_chart()
        # Begin draining the status queue on the main thread every 100 ms
        self._poll_status_queue()

    # ══════════════════════════════════════════════════════════════════════════
    # UI Construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self._build_header()
        self._build_tabs()

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=C["surface"], corner_radius=0, height=72)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr,
            text="◈  SENTIMENT ANALYZER  PRO",
            font=ctk.CTkFont(family="Courier New", size=17, weight="bold"),
            text_color=C["accent"],
        ).place(relx=0.04, rely=0.5, anchor="w")

        self.status_dot = ctk.CTkLabel(
            hdr, text="● Loading…",
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color=C["medium"],
        )
        self.status_dot.place(relx=0.97, rely=0.5, anchor="e")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    def _build_tabs(self):
        self.tabs = ctk.CTkTabview(
            self,
            fg_color=C["bg"],
            segmented_button_fg_color=C["surface"],
            segmented_button_selected_color=C["accent2"],
            segmented_button_selected_hover_color=C["accent"],
            segmented_button_unselected_color=C["surface"],
            segmented_button_unselected_hover_color=C["tile"],
            text_color=C["text"],
            text_color_disabled=C["muted"],
            border_color=C["border"],
        )
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        for name in ("Analyze", "Batch", "History", "Insights"):
            self.tabs.add(name)

        self._build_analyze_tab(self.tabs.tab("Analyze"))
        self._build_batch_tab(self.tabs.tab("Batch"))
        self._build_history_tab(self.tabs.tab("History"))
        self._build_insights_tab(self.tabs.tab("Insights"))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — ANALYZE
    # ══════════════════════════════════════════════════════════════════════════

    def _build_analyze_tab(self, parent):
        scroll = ctk.CTkScrollableFrame(
            parent,
            fg_color=C["bg"],
            scrollbar_button_color=C["border"],
            scrollbar_button_hover_color=C["accent"],
        )
        scroll.pack(fill="both", expand=True)

        self._build_input_section(scroll)
        self._build_action_buttons(scroll)
        self._build_result_section(scroll)
        self._build_model_scores(scroll)
        self._build_emotion_section(scroll)
        self._build_aspect_section(scroll)
        self._build_translation_section(scroll)

    # ── Text input ────────────────────────────────────────────────────────────
    def _build_input_section(self, p):
        self._section(p, "TEXT INPUT")

        meta_row = ctk.CTkFrame(p, fg_color="transparent")
        meta_row.pack(fill="x", pady=(0, 4))

        self.char_count = ctk.CTkLabel(
            meta_row, text="0 chars  ·  0 words",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C["muted"],
        )
        self.char_count.pack(side="right")

        self.negation_badge = ctk.CTkLabel(
            meta_row, text="",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C["medium"],
        )
        self.negation_badge.pack(side="right", padx=12)

        self.text_input = ctk.CTkTextbox(
            p, height=110,
            fg_color=C["tile"], text_color=C["text"],
            border_color=C["border"], border_width=1,
            corner_radius=10,
            font=ctk.CTkFont(family="Courier New", size=13),
            wrap="word",
        )
        self.text_input.pack(fill="x", pady=(0, 14))
        self.text_input.insert("0.0", "Type or speak your text here…")
        self.text_input.bind("<FocusIn>",  self._clear_placeholder)
        self.text_input.bind("<KeyRelease>", self._update_char_count)

    # ── Buttons ───────────────────────────────────────────────────────────────
    def _build_action_buttons(self, p):
        row = ctk.CTkFrame(p, fg_color="transparent")
        row.pack(fill="x", pady=(0, 16))

        B = dict(
            corner_radius=8, height=40,
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
        )

        self.analyze_btn = ctk.CTkButton(
            row, text="⬡  ANALYZE",
            fg_color=C["accent2"], hover_color=C["accent"],
            text_color=C["bg"], command=self._on_analyze, **B
        )
        self.analyze_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.voice_btn = ctk.CTkButton(
            row, text="⏺  VOICE",
            fg_color=C["tile"], hover_color=C["border"],
            border_width=1, border_color=C["accent"],
            text_color=C["accent"], command=self._on_voice, **B
        )
        self.voice_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.export_btn = ctk.CTkButton(
            row, text="↓  EXPORT",
            fg_color=C["tile"], hover_color=C["border"],
            border_width=1, border_color=C["purple"],
            text_color=C["purple"], command=self._on_export_session, **B
        )
        self.export_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.clear_btn = ctk.CTkButton(
            row, text="✕",
            fg_color=C["tile"], hover_color=C["border"],
            border_width=1, border_color=C["border"],
            text_color=C["muted"], command=self._on_clear,
            width=44, **B
        )
        self.clear_btn.pack(side="left")

    # ── Result card ───────────────────────────────────────────────────────────
    def _build_result_section(self, p):
        self._section(p, "SENTIMENT RESULT")

        self.result_frame = ctk.CTkFrame(
            p, fg_color=C["tile"], border_color=C["border"],
            border_width=1, corner_radius=12, height=108,
        )
        self.result_frame.pack(fill="x", pady=(4, 14))
        self.result_frame.pack_propagate(False)

        self.result_label = ctk.CTkLabel(
            self.result_frame, text="—",
            font=ctk.CTkFont(family="Courier New", size=40, weight="bold"),
            text_color=C["muted"],
        )
        self.result_label.place(relx=0.5, rely=0.40, anchor="center")

        self.result_meta = ctk.CTkLabel(
            self.result_frame, text="run analysis to see results",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C["muted"],
        )
        self.result_meta.place(relx=0.5, rely=0.78, anchor="center")

    # ── Model score panel ─────────────────────────────────────────────────────
    def _build_model_scores(self, p):
        self._section(p, "MODEL SCORES  (3-model ensemble)")

        outer = ctk.CTkFrame(
            p, fg_color=C["tile"], border_color=C["border"],
            border_width=1, corner_radius=12,
        )
        outer.pack(fill="x", pady=(4, 14))

        # VADER column
        vc = ctk.CTkFrame(outer, fg_color="transparent")
        vc.pack(side="left", expand=True, fill="both", padx=14, pady=10)
        self._mini_label(vc, "VADER")
        self.vader_compound = ctk.CTkLabel(
            vc, text="—",
            font=ctk.CTkFont(family="Courier New", size=22, weight="bold"),
            text_color=C["accent"],
        )
        self.vader_compound.pack(anchor="w")
        self.vader_detail = ctk.CTkLabel(
            vc, text="pos / neu / neg",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C["muted"],
        )
        self.vader_detail.pack(anchor="w")

        self._vdiv(outer)

        # RoBERTa column
        rc = ctk.CTkFrame(outer, fg_color="transparent")
        rc.pack(side="left", expand=True, fill="both", padx=14, pady=10)
        self._mini_label(rc, "RoBERTa")
        self.rob_bars = {}
        for lbl in LABELS:
            self._prob_bar_row(rc, lbl, self.rob_bars)

        self._vdiv(outer)

        # DistilBERT column
        dc = ctk.CTkFrame(outer, fg_color="transparent")
        dc.pack(side="left", expand=True, fill="both", padx=14, pady=10)
        self._mini_label(dc, "DistilBERT")
        self.dbert_label = ctk.CTkLabel(
            dc, text="—",
            font=ctk.CTkFont(family="Courier New", size=18, weight="bold"),
            text_color=C["accent"],
        )
        self.dbert_label.pack(anchor="w")
        self.dbert_score = ctk.CTkLabel(
            dc, text="score: —",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C["muted"],
        )
        self.dbert_score.pack(anchor="w")

        self._vdiv(outer)

        # Ensemble column
        ec = ctk.CTkFrame(outer, fg_color="transparent")
        ec.pack(side="left", expand=True, fill="both", padx=14, pady=10)
        self._mini_label(ec, "ENSEMBLE")
        self.ens_bars = {}
        for lbl in LABELS:
            self._prob_bar_row(ec, lbl, self.ens_bars)

    # ── Emotion wheel ─────────────────────────────────────────────────────────
    def _build_emotion_section(self, p):
        self._section(p, "EMOTION ANALYSIS  (7 classes)")

        emo_frame = ctk.CTkFrame(
            p, fg_color=C["tile"], border_color=C["border"],
            border_width=1, corner_radius=12,
        )
        emo_frame.pack(fill="x", pady=(4, 14))

        self.emo_bars: dict[str, tuple] = {}
        for emo in ("joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"):
            row = ctk.CTkFrame(emo_frame, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=2)

            ctk.CTkLabel(
                row, text=emo.capitalize(),
                width=68,
                font=ctk.CTkFont(family="Courier New", size=11),
                text_color=EMOTION_COLORS[emo],
            ).pack(side="left")

            bar = ctk.CTkProgressBar(
                row, height=8, corner_radius=4,
                fg_color=C["surface"],
                progress_color=EMOTION_COLORS[emo],
            )
            bar.set(0)
            bar.pack(side="left", fill="x", expand=True, padx=(4, 8))

            pct = ctk.CTkLabel(
                row, text="0%", width=38,
                font=ctk.CTkFont(family="Courier New", size=10),
                text_color=C["muted"],
            )
            pct.pack(side="left")
            self.emo_bars[emo] = (bar, pct)

    # ── Aspect section ────────────────────────────────────────────────────────
    def _build_aspect_section(self, p):
        self._section(p, "KEY ASPECTS DETECTED")

        self.aspect_frame = ctk.CTkFrame(
            p, fg_color=C["tile"], border_color=C["border"],
            border_width=1, corner_radius=12, height=48,
        )
        self.aspect_frame.pack(fill="x", pady=(4, 14))
        self.aspect_frame.pack_propagate(False)

        self.aspect_label = ctk.CTkLabel(
            self.aspect_frame,
            text="aspects will appear here",
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color=C["muted"],
        )
        self.aspect_label.place(relx=0.5, rely=0.5, anchor="center")

    # ── Translation ───────────────────────────────────────────────────────────
    def _build_translation_section(self, p):
        self._section(p, "TRANSLATE")

        t_frame = ctk.CTkFrame(
            p, fg_color=C["tile"], border_color=C["border"],
            border_width=1, corner_radius=12,
        )
        t_frame.pack(fill="x", pady=(4, 14))

        inner = ctk.CTkFrame(t_frame, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=10)

        self.lang_var = StringVar(value="Select language")
        lang_names = sorted(v.title() for v in GOOGLE_LANGUAGES_TO_CODES.keys())

        self.lang_menu = ctk.CTkOptionMenu(
            inner, variable=self.lang_var, values=lang_names,
            fg_color=C["surface"], button_color=C["accent2"],
            button_hover_color=C["accent"], dropdown_fg_color=C["surface"],
            text_color=C["text"],
            font=ctk.CTkFont(family="Courier New", size=12),
            width=220, dynamic_resizing=False,
        )
        self.lang_menu.pack(side="left")

        ctk.CTkButton(
            inner, text="TRANSLATE →",
            fg_color=C["surface"], hover_color=C["border"],
            border_width=1, border_color=C["accent2"], text_color=C["accent2"],
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            corner_radius=8, height=34, command=self._on_translate,
        ).pack(side="left", padx=(8, 0))

        self.translation_box = ctk.CTkTextbox(
            t_frame, height=80,
            fg_color=C["surface"], text_color=C["text"],
            border_color=C["border"], border_width=1, corner_radius=8,
            font=ctk.CTkFont(family="Courier New", size=13),
            state="disabled", wrap="word",
        )
        self.translation_box.pack(fill="x", padx=14, pady=(0, 12))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — BATCH
    # ══════════════════════════════════════════════════════════════════════════

    def _build_batch_tab(self, parent):
        self._section(parent, "BATCH ANALYSIS  (upload CSV)")

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=8)

        ctk.CTkButton(
            row, text="📂  Open CSV",
            fg_color=C["accent2"], hover_color=C["accent"],
            text_color=C["bg"],
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            corner_radius=8, height=38,
            command=self._on_batch_open,
        ).pack(side="left", padx=(0, 10))

        self.batch_col_var = StringVar(value="text")
        ctk.CTkLabel(
            row, text="column:",
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color=C["muted"],
        ).pack(side="left")
        self.batch_col_entry = ctk.CTkEntry(
            row, textvariable=self.batch_col_var,
            width=120, fg_color=C["tile"],
            border_color=C["border"], text_color=C["text"],
            font=ctk.CTkFont(family="Courier New", size=11),
        )
        self.batch_col_entry.pack(side="left", padx=(4, 0))

        self.batch_progress = ctk.CTkProgressBar(
            parent, height=10, corner_radius=5,
            fg_color=C["surface"], progress_color=C["accent2"],
        )
        self.batch_progress.set(0)
        self.batch_progress.pack(fill="x", pady=10)

        self.batch_status = ctk.CTkLabel(
            parent, text="no file loaded",
            font=ctk.CTkFont(family="Courier New", size=11),
            text_color=C["muted"],
        )
        self.batch_status.pack(anchor="w")

        self.batch_box = ctk.CTkTextbox(
            parent, height=360,
            fg_color=C["tile"], text_color=C["text"],
            border_color=C["border"], border_width=1, corner_radius=10,
            font=ctk.CTkFont(family="Courier New", size=11),
            state="disabled",
        )
        self.batch_box.pack(fill="both", expand=True, pady=(8, 0))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — HISTORY
    # ══════════════════════════════════════════════════════════════════════════

    def _build_history_tab(self, parent):
        toolbar = ctk.CTkFrame(parent, fg_color="transparent")
        toolbar.pack(fill="x", pady=(4, 8))

        ctk.CTkButton(
            toolbar, text="⟳  Refresh",
            fg_color=C["tile"], hover_color=C["border"],
            border_width=1, border_color=C["border"],
            text_color=C["accent"],
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            corner_radius=6, height=30, width=90,
            command=self._refresh_history,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            toolbar, text="↓ CSV",
            fg_color=C["tile"], hover_color=C["border"],
            border_width=1, border_color=C["purple"],
            text_color=C["purple"],
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            corner_radius=6, height=30, width=70,
            command=lambda: self._export_db("csv"),
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            toolbar, text="↓ JSON",
            fg_color=C["tile"], hover_color=C["border"],
            border_width=1, border_color=C["teal"],
            text_color=C["teal"],
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            corner_radius=6, height=30, width=70,
            command=lambda: self._export_db("json"),
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            toolbar, text="🗑 Clear DB",
            fg_color=C["tile"], hover_color=C["border"],
            border_width=1, border_color=C["high"],
            text_color=C["high"],
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            corner_radius=6, height=30, width=90,
            command=self._clear_db,
        ).pack(side="left")

        self.history_box = ctk.CTkTextbox(
            parent, height=500,
            fg_color=C["tile"], text_color=C["text"],
            border_color=C["border"], border_width=1, corner_radius=10,
            font=ctk.CTkFont(family="Courier New", size=11),
            state="disabled",
        )
        self.history_box.pack(fill="both", expand=True)
        self._refresh_history()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — INSIGHTS (live charts)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_insights_tab(self, parent):
        self._section(parent, "SENTIMENT DISTRIBUTION")

        chart_frame = ctk.CTkFrame(
            parent, fg_color=C["tile"], border_color=C["border"],
            border_width=1, corner_radius=12, height=260,
        )
        chart_frame.pack(fill="x", pady=(4, 14))

        self.dist_chart = EmbeddedChart(chart_frame, figsize=(7, 2.4))

        self._section(parent, "CONFIDENCE OVER TIME")

        conf_frame = ctk.CTkFrame(
            parent, fg_color=C["tile"], border_color=C["border"],
            border_width=1, corner_radius=12, height=240,
        )
        conf_frame.pack(fill="x", pady=(4, 14))

        self.conf_chart = EmbeddedChart(conf_frame, figsize=(7, 2.2))

        ctk.CTkButton(
            parent, text="⟳  Refresh Charts",
            fg_color=C["tile"], hover_color=C["border"],
            border_width=1, border_color=C["accent"],
            text_color=C["accent"],
            font=ctk.CTkFont(family="Courier New", size=12, weight="bold"),
            corner_radius=8, height=36,
            command=self._refresh_stats_chart,
        ).pack(pady=4)

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _section(self, parent, text: str):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=C["muted"], anchor="w",
        ).pack(anchor="w", pady=(10, 2))

    def _mini_label(self, parent, text: str):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
            text_color=C["muted"],
        ).pack(anchor="w")

    def _vdiv(self, parent):
        ctk.CTkFrame(parent, fg_color=C["border"], width=1).pack(
            side="left", fill="y", pady=8
        )

    def _prob_bar_row(self, parent, label: str, store: dict):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(
            row, text=label[:3].upper(), width=28,
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C["muted"],
        ).pack(side="left")
        bar = ctk.CTkProgressBar(
            row, height=8, corner_radius=4,
            fg_color=C["surface"],
            progress_color=SC[label][0],
        )
        bar.set(0)
        bar.pack(side="left", fill="x", expand=True, padx=(4, 6))
        pct = ctk.CTkLabel(
            row, text="0%", width=34,
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=C["muted"],
        )
        pct.pack(side="left")
        store[label] = (bar, pct)

    def _set_status(self, msg: str):
        """Called ONLY from the main thread (via _poll_status_queue)."""
        self.status_dot.configure(
            text=f"● {msg}",
            text_color=C["low"] if "ready" in msg.lower() else C["medium"],
        )

    def _poll_status_queue(self):
        """Drain the thread-safe status queue every 100 ms on the main thread."""
        try:
            while True:
                msg = self._status_queue.get_nowait()
                self._set_status(msg)
        except queue.Empty:
            pass
        # Reschedule — runs forever on the main thread
        self.after(100, self._poll_status_queue)

    def _clear_placeholder(self, _=None):
        if self.text_input.get("0.0", "end").strip() == "Type or speak your text here…":
            self.text_input.delete("0.0", "end")

    def _get_input(self) -> str:
        return self.text_input.get("0.0", "end-1c").strip()

    def _update_char_count(self, _=None):
        txt = self._get_input()
        chars = len(txt)
        words = len(txt.split()) if txt else 0
        self.char_count.configure(text=f"{chars} chars  ·  {words} words")

    def _flash(self, msg: str, error: bool = False):
        color = C["high"] if error else C["low"]
        self.status_dot.configure(text=f"● {msg}", text_color=color)
        self.after(3500, lambda: self.status_dot.configure(
            text="● Ready" if self.engine.ready else "● Loading…",
            text_color=C["low"] if self.engine.ready else C["medium"],
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # Analysis event handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _on_analyze(self):
        text = self._get_input()
        if not text or text == "Type or speak your text here…":
            self._flash("Please enter some text first.", error=True); return
        if not self.engine.ready:
            self._flash("Models still loading — please wait.", error=True); return

        self.analyze_btn.configure(state="disabled", text="⏳  ANALYZING…")
        threading.Thread(target=self._run_analysis, args=(text,), daemon=True).start()

    def _run_analysis(self, text: str):
        try:
            result = self.engine.analyze(text, source="typed")
            self.after(0, self._display_result, result)
        except Exception as exc:
            self.after(0, self._flash, str(exc), True)
        finally:
            self.after(0, lambda: self.analyze_btn.configure(
                state="normal", text="⬡  ANALYZE"
            ))

    def _display_result(self, r: dict):
        label = r["final"]
        fg, bg = SC[label]
        conf   = r["confidence"]

        # Result card
        self.result_frame.configure(fg_color=bg, border_color=fg)
        self.result_label.configure(text=label.upper(), text_color=fg)

        models_agree = "✓ all 3 agree" if r["agreed"] else "⚡ ensemble override"
        neg_txt      = "  ·  negation detected" if r["negation"] else ""
        self.result_meta.configure(
            text=f"confidence {conf:.0%}  ·  {models_agree}{neg_txt}",
            text_color=fg,
        )

        # Negation badge
        self.negation_badge.configure(
            text="⚠ negation" if r["negation"] else ""
        )

        # VADER
        v = r["vader"]
        self.vader_compound.configure(text=f"{v['compound']:+.3f}", text_color=fg)
        s = v["scores"]
        self.vader_detail.configure(
            text=f"pos {s['pos']:.2f}  neu {s['neu']:.2f}  neg {s['neg']:.2f}"
        )

        # RoBERTa bars
        for lbl, (bar, pct) in self.rob_bars.items():
            val = r["roberta"]["probs"].get(lbl, 0.0)
            bar.set(val); pct.configure(text=f"{val:.0%}")

        # DistilBERT
        d = r["distilbert"]
        d_fg = SC.get(d["label"], (C["accent"], ""))[0]
        self.dbert_label.configure(text=d["label"], text_color=d_fg)
        self.dbert_score.configure(text=f"score: {d['score']:.3f}")

        # Ensemble bars
        for lbl, (bar, pct) in self.ens_bars.items():
            val = r["ensemble_probs"].get(lbl, 0.0)
            bar.set(val); pct.configure(text=f"{val:.0%}")

        # Emotions
        for emo, (bar, pct) in self.emo_bars.items():
            val = r["emotions"].get(emo, 0.0)
            bar.set(val); pct.configure(text=f"{val:.0%}")

        # Aspects
        aspects = r.get("aspects", [])
        self.aspect_label.configure(
            text="  ·  ".join(aspects) if aspects else "no noun aspects detected",
            text_color=C["teal"] if aspects else C["muted"],
        )

        # Persist + session
        self._save_to_db(r)
        self._session_results.append(r)

    def _save_to_db(self, r: dict):
        emo = r.get("emotions", {})
        try:
            self.db.insert({
                "ts":               r["ts"],
                "original":         r["original"],
                "normalized":       r["normalized"],
                "final_label":      r["final"],
                "confidence":       r["confidence"],
                "vader_compound":   r["vader"]["compound"],
                "roberta_neg":      r["roberta"]["probs"]["Negative"],
                "roberta_neu":      r["roberta"]["probs"]["Neutral"],
                "roberta_pos":      r["roberta"]["probs"]["Positive"],
                "distilbert_label": r["distilbert"]["label"],
                "distilbert_score": r["distilbert"]["score"],
                "emotions":         json.dumps(emo),
                "aspects":          json.dumps(r.get("aspects", [])),
                "agreed":           int(r["agreed"]),
                "source":           r.get("source", "typed"),
                "lang_detected":    "",
            })
        except Exception as e:
            print(f"DB write error: {e}")

    # ── Voice ─────────────────────────────────────────────────────────────────
    def _on_voice(self):
        if self.is_listening: return
        self.is_listening = True
        self.voice_btn.configure(
            text="⏹  LISTENING…", fg_color=C["high"], text_color=C["bg"]
        )
        threading.Thread(target=self._run_voice, daemon=True).start()

    def _run_voice(self):
        try:
            text = self.voice_cap.capture()
            self.after(0, lambda: (
                self.text_input.delete("0.0", "end"),
                self.text_input.insert("0.0", text),
                self._flash("Voice captured ✓"),
                self._update_char_count(),
            ))
        except sr.WaitTimeoutError:
            self.after(0, self._flash, "Timed out — no speech detected.", True)
        except sr.UnknownValueError:
            self.after(0, self._flash, "Could not understand audio.", True)
        except sr.RequestError as e:
            self.after(0, self._flash, f"Speech API error: {e}", True)
        except Exception as e:
            self.after(0, self._flash, str(e), True)
        finally:
            self.after(0, self._reset_voice_btn)

    def _reset_voice_btn(self):
        self.is_listening = False
        self.voice_btn.configure(
            text="⏺  VOICE", fg_color=C["tile"], text_color=C["accent"]
        )

    # ── Clear ─────────────────────────────────────────────────────────────────
    def _on_clear(self):
        self.text_input.delete("0.0", "end")
        self.result_label.configure(text="—", text_color=C["muted"])
        self.result_meta.configure(
            text="run analysis to see results", text_color=C["muted"]
        )
        self.result_frame.configure(
            fg_color=C["tile"], border_color=C["border"]
        )
        self.vader_compound.configure(text="—", text_color=C["accent"])
        self.vader_detail.configure(text="pos / neu / neg")
        for _, (bar, pct) in {**self.rob_bars, **self.ens_bars}.items():
            bar.set(0); pct.configure(text="0%")
        for _, (bar, pct) in self.emo_bars.items():
            bar.set(0); pct.configure(text="0%")
        self.dbert_label.configure(text="—", text_color=C["accent"])
        self.dbert_score.configure(text="score: —")
        self.aspect_label.configure(
            text="aspects will appear here", text_color=C["muted"]
        )
        self.char_count.configure(text="0 chars  ·  0 words")
        self.negation_badge.configure(text="")
        self.translation_box.configure(state="normal")
        self.translation_box.delete("0.0", "end")
        self.translation_box.configure(state="disabled")

    # ── Translate ─────────────────────────────────────────────────────────────
    def _on_translate(self):
        text = self._get_input()
        lang_name = self.lang_var.get()
        if not text or text == "Type or speak your text here…":
            self._flash("Enter text before translating.", error=True); return
        if lang_name == "Select language":
            self._flash("Choose a target language.", error=True); return
        lang_code = self._lang_map.get(lang_name)
        if not lang_code:
            self._flash("Unknown language.", error=True); return

        threading.Thread(
            target=self._run_translation, args=(text, lang_code), daemon=True
        ).start()

    def _run_translation(self, text: str, lang_code: str):
        try:
            translated = self.engine.translate(text, lang_code)
            def _update():
                self.translation_box.configure(state="normal")
                self.translation_box.delete("0.0", "end")
                self.translation_box.insert("0.0", translated)
                self.translation_box.configure(state="disabled")
                self._flash("Translation complete ✓")
            self.after(0, _update)
        except Exception as e:
            self.after(0, self._flash, f"Translation failed: {e}", True)

    # ── Batch ─────────────────────────────────────────────────────────────────
    def _on_batch_open(self):
        if not self.engine.ready:
            self._flash("Models still loading.", error=True); return
        path = filedialog.askopenfilename(
            title="Open CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path: return
        col = self.batch_col_var.get().strip() or "text"
        self.batch_status.configure(text=f"Reading {Path(path).name} …")
        threading.Thread(
            target=self._run_batch, args=(path, col), daemon=True
        ).start()

    def _run_batch(self, path: str, col: str):
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows   = [r for r in reader]

            if not rows:
                self.after(0, lambda: self.batch_status.configure(
                    text="CSV is empty."
                ))
                return

            if col not in rows[0]:
                available = ", ".join(rows[0].keys())
                self.after(0, lambda: self.batch_status.configure(
                    text=f"Column '{col}' not found. Available: {available}"
                ))
                return

            texts = [r[col] for r in rows if r.get(col)]
            self.after(0, lambda: self.batch_status.configure(
                text=f"Analyzing {len(texts)} rows…"
            ))
            self.after(0, lambda: self.batch_progress.set(0))

            results = self.engine.batch_analyze(
                texts,
                progress_cb=lambda p: self.after(
                    0, lambda p=p: self.batch_progress.set(p)
                ),
                source="batch",
            )

            # Save all to DB
            for r in results:
                if "error" not in r:
                    self._save_to_db(r)

            # Summary text
            labels  = [r.get("final", "Error") for r in results]
            counter = Counter(labels)
            summary = "\n".join(
                f"  {lbl:<10} {cnt:>4}  ({cnt/len(labels):.0%})"
                for lbl, cnt in sorted(counter.items())
            )
            lines = [
                f"═══ BATCH RESULTS — {len(results)} rows ═══",
                "",
                summary,
                "",
                "─── Top 10 samples ───",
            ]
            for r in results[:10]:
                if "error" in r:
                    lines.append(f"  ERROR: {r['original'][:50]}")
                else:
                    lines.append(
                        f"  [{r['final']:<8}] {r['confidence']:.0%}  "
                        f"{'[NEG]' if r['negation'] else '     '}  "
                        f"{r['original'][:60]}"
                    )

            def _show():
                self.batch_box.configure(state="normal")
                self.batch_box.delete("0.0", "end")
                self.batch_box.insert("0.0", "\n".join(lines))
                self.batch_box.configure(state="disabled")
                self.batch_status.configure(
                    text=f"Done — {len(results)} rows analyzed."
                )
                self.batch_progress.set(1.0)

            self.after(0, _show)
            self.after(0, self._refresh_stats_chart)

        except Exception as e:
            self.after(0, lambda: self.batch_status.configure(
                text=f"Error: {e}"
            ))
            traceback.print_exc()

    # ── Export session ────────────────────────────────────────────────────────
    def _on_export_session(self):
        if not self._session_results:
            self._flash("No results in current session yet.", error=True); return

        path = filedialog.asksaveasfilename(
            title="Save session results",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv")],
        )
        if not path: return

        if path.endswith(".csv"):
            cols = ["ts", "original", "final", "confidence",
                    "agreed", "negation", "source"]
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                w.writerows(self._session_results)
        else:
            # JSON — strip non-serialisable objects
            safe = []
            for r in self._session_results:
                safe.append({k: v for k, v in r.items()
                              if isinstance(v, (str, int, float, bool, list, dict))})
            with open(path, "w", encoding="utf-8") as f:
                json.dump(safe, f, indent=2, ensure_ascii=False)

        self._flash(f"Exported {len(self._session_results)} results ✓")

    # ── History tab ───────────────────────────────────────────────────────────
    def _refresh_history(self):
        rows = self.db.fetch_all(limit=100)
        lines = []
        for r in rows:
            label = r["final_label"]
            conf  = r.get("confidence", 0)
            src   = r.get("source", "")
            neg   = "[NEG]" if r.get("negation") else "     "
            ts    = r["ts"][:19]
            snippet = r["original"][:70].replace("\n", " ")
            lines.append(
                f"{ts}  [{label:<8}]  {conf:.0%}  {neg}  ({src[:5]})  {snippet}"
            )

        self.history_box.configure(state="normal")
        self.history_box.delete("0.0", "end")
        self.history_box.insert(
            "0.0",
            "\n".join(lines) if lines else "no history yet"
        )
        self.history_box.configure(state="disabled")

    def _export_db(self, fmt: str):
        path = filedialog.asksaveasfilename(
            title="Export history",
            defaultextension=f".{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}")],
        )
        if not path: return
        if fmt == "csv":
            self.db.export_csv(path)
        else:
            self.db.export_json(path)
        self._flash(f"History exported as {fmt.upper()} ✓")

    def _clear_db(self):
        if messagebox.askyesno("Clear Database", "Delete all history?"):
            self.db.clear()
            self._refresh_history()
            self._flash("Database cleared.")

    # ── Insights charts ───────────────────────────────────────────────────────
    def _refresh_stats_chart(self):
        stats = self.db.stats()
        rows  = self.db.fetch_all(limit=50)

        # ── Distribution bar chart ──────────────────
        self.dist_chart.clear()
        ax = self.dist_chart.ax
        if stats:
            labels_ = list(stats.keys())
            counts  = [stats[l]["count"] for l in labels_]
            colors_ = [SC.get(l, (C["muted"], ""))[0] for l in labels_]
            bars    = ax.bar(labels_, counts, color=colors_, width=0.45,
                             edgecolor="none")
            for bar, cnt in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(cnt),
                    ha="center", va="bottom", color=C["text"],
                    fontsize=9, fontfamily="monospace"
                )
            ax.set_title("Sentiment Distribution", color=C["text"],
                         fontsize=10, fontfamily="monospace", pad=6)
        else:
            ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                    transform=ax.transAxes, color=C["muted"], fontsize=10)

        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        ax.tick_params(colors=C["muted"], labelsize=9)
        ax.yaxis.set_tick_params(labelcolor=C["muted"])
        self.dist_chart.draw()

        # ── Confidence over time ─────────────────────
        self.conf_chart.clear()
        ax2 = self.conf_chart.ax
        if rows:
            confs  = [r.get("confidence", 0) for r in reversed(rows)]
            colors_ = [SC.get(r["final_label"], (C["muted"], ""))[0]
                       for r in reversed(rows)]
            ax2.scatter(range(len(confs)), confs, c=colors_,
                        s=20, alpha=0.85, linewidths=0)
            ax2.plot(confs, color=C["muted"], linewidth=0.6, alpha=0.5)
            ax2.set_ylim(0, 1)
            ax2.set_title("Confidence over Time (recent 50)",
                          color=C["text"], fontsize=10,
                          fontfamily="monospace", pad=6)
            ax2.axhline(0.5, color=C["border"], linewidth=0.8, linestyle="--")
        else:
            ax2.text(0.5, 0.5, "no data yet", ha="center", va="center",
                     transform=ax2.transAxes, color=C["muted"], fontsize=10)

        for spine in ax2.spines.values():
            spine.set_edgecolor(C["border"])
        ax2.tick_params(colors=C["muted"], labelsize=9)
        self.conf_chart.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()