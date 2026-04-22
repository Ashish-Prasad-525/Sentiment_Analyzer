Built a multi-model NLP desktop app (VADER + RoBERTa + DistilBERT ensemble) achieving weighted confidence scoring, negation-aware calibration, and 7-class emotion analysis.

Designed a SQLite persistence layer with thread-safe writes, full CRUD, and CSV/JSON export; embedded live matplotlib charts (sentiment distribution, confidence over time) into the GUI.

Implemented a CSV batch-processing pipeline supporting thousands of rows with real-time progress reporting and asynchronous multi-threaded execution to keep the UI responsive.

Applied NLP text preprocessing including emoji-to-word mapping, internet slang normalisation,regex negation detection, and NLTK POS-tag-based aspect extraction.

Features :
Analyze tab: type or speak → instant 3-model result + emotion radar + aspect chips
Batch tab: upload any CSV, pick the text column, run in background, export results
History tab: every analysis persisted in SQLite; export full DB as CSV or JSON
Insights tab: auto-updating bar chart (sentiment distribution) + scatter/line chart (confidence over time)
