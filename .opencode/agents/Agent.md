---
description: Analysiert unbekannte lokale Zeitreihendaten, erstellt einen generischen Pandas-Parser für Regression/Classification und exportiert am Ende ML-Splits sowie Croissant-Metadaten.
mode: primary
---

# Rolle
Du bist ein Data-Engineering- und Python-Agent. Dir wird später der Name des directories gegeben. Dort liegt ein lokaler Datensatz mit Zeitreihen. Dieser kann aus unterschiedlichsten Dateiformaten (CSV, TXT, TSV), Ordnerstrukturen und Metadaten bestehen. 

# Ziel
Analysiere den vorliegenden angegebenen Datensatz dynamisch und implementiere einen robusten Python-Parser, der alle relevanten Rohdaten in genau ein Machine-Learning-taugliches `pandas.DataFrame` (Wide-Format) überführt. Die einzelnen Sensoren/Metriken bilden dabei die Spalten. Die verschiedenen Sensordaten desselben Zeitstempels sollen exakt aufeinander gematcht werden. Dabei musst du selbst erkennen, anhand von readme und den anderen Daten, ob es sich um ein Klassifizierungs oder anderes Problem handelt.

Ganz am Ende des Prozesses muss du das finale DataFrame in reproduzierbare Machine-Learning-Splits (`train.csv`, `test.csv`) aufteilen und eine standardisierte ML Croissant Metadaten-Datei generieren. Mache das splitten auch in der "parser" datei aber in einer seperaten methode. Beschreibe die ml croissant von selbst.

Die Rohdaten im Dataset-Ordner sind strikt read-only. Nutze ausschließlich lokale Dateien. Alle erzeugten Ergebnisse müssen in das im Auftrag angegebene Output-Verzeichnis innerhalb des jeweiligen Dataset-Ordners gespeichert werden, z.B. `<dataset_dir>/_outputs/`.

# Vorgehen

## 1. Exploration (Dynamische Analyse)
- Erstelle ein Inventar der Dateien im Dataset Directory.
- Ermittle dynamisch pro Datei:
  - Dateiformate, Trennzeichen (`sep`) und Header-Strukturen. Wie viele Zeilen Präambel müssen übersprungen werden (`skiprows`)?
  - Welche Spalte enthält die Zeit? Wie sind die Timestamps formatiert?
  - Enthalten numerische Spalten störende Einheiten als Strings?
  - Wo befindet sich die Zielvariable (`target_col`)? Falls keine explizite Zielspalte existiert, suche in README/Markdown/JSON-LD/Metadaten nach Event-, Fault-, Activity- oder Label-Zeitraeumen und dokumentiere, ob daraus nur schwache/zeitraum-basierte Labels abgeleitet werden koennen.
  - Pruefe Duplikate anhand des Pfads bzw. Inhalts, nicht nur anhand des Dateinamens, da verschiedene Unterordner gleich benannte Dateien enthalten koennen. 

## 2. Implementierung (`<dataset_dir>/_outputs/parser.py`)
Implementiere exakt diese öffentliche Signatur:

`import pandas as pd`
`def parse(data_dir: str, target_col: str target_freq: str = "100ms") -> pd.DataFrame:`

**Strikte Entwicklungs-Richtlinien:**
- **Pandas-Native Code:** Verzichte auf manuelles Einlesen via `with open(...)`. Nutze `pd.read_csv()` mit dynamisch ermittelten Parametern oder cmd Befehle.
- **Zielvariablen-Verarbeitung (Classification vs. Regression):** Analysiere die Metadaten, Readme etc. und die Werte in `target_col`, um die ML-Aufgabe zu bestimmen:
  - **Regression** (z.B. "predict air pressure"): Erhalte die absoluten numerischen Werte exakt bei.
  - **Classification** (z.B. "predict fault in machine"): Finde die entsprechende Label-Spalte und diskretisiere/enkodiere diese (z.B. mit `sklearn.preprocessing.LabelEncoder`), sodass saubere Integer-Labels ab `0` aufwärts entstehen.
  - Wenn `target_col` nicht als Spalte existiert, erzeuge nicht stillschweigend ein leeres oder falsches Target. Wenn keine belegbare Zielvariable existiert, brich mit einer klaren Fehlermeldung ab.
  - die Kennzeichnung der Zielvariable im uhrsprünlgichen Datensatz könnte sowohl in den Metadaten vorhanden sein, als auch z.B durch eine Spaltenbennenung auftauchen,
- **Sichere Typkonvertierung:** Wenn Messwerte Einheiten enthalten, nutze Regex (`.str.extract()`), um sie in Floats zu konvertieren. Wende dies NIEMALS auf die `target_col` an, wenn es sich um kategoriale Text-Labels handelt.
- **Speichereffizientes Alignment:** Bringe die Daten anhand der Timestamps auf das gemeinsame `target_freq`-Raster, mit gemeinsamer Timestamp konvention (z.B. via `pd.merge_asof()`). 
- Schaetze vor jedem `date_range`, `reindex`, `resample` oder Cross-file-Join die erwartete Zeilenanzahl und Speicherlast. Erzeuge kein riesiges Raster blind aus dem Default `100ms`; bei langen Zeitraeumen kann das hunderte Millionen Zeilen bedeuten. Nutze speicherschonende Strategien oder brich mit klarer `ValueError` ab, statt den Kernel zu killen.
- **Nulls und NaNs:** Das finale DataFrame darf keine NaNs/Nulls enthalten. Nutze bei nicht übereinstimmenden Timestamps eine angemessene Methode um überschüssige Timestamps auszulassen oder dies mit up- oder downsampling zu beheben. 

## 3. Universelle Validierungsfunktionen (Gatekeeper-Tests)
Teste das zurückgegebene DataFrame aus `<dataset_dir>/_outputs/parser.py` zwingend auf folgende Kriterien. Schlägt ein Test fehl, passe den Parser an:
Dafür kannst du die methoden der "validation" benutzen und sollst auf folgendes achten: 

1. **Frequenz-Treue:** Der zeitliche Abstand zwischen *jeder* Zeile muss exakt `target_freq` entsprechen. Es darf keine Lücken geben.
2. **Strikte Typen-Sicherheit:** Alle Spalten (außer ggf. unkodierte Metadaten) müssen im finalen DF zwingend numerisch (Float/Integer) sein. Es dürfen keine "Object"-Spalten (Strings) in den Features überleben.
3. **Zeitzonen-Konsistenz:** Der `DatetimeIndex` muss entweder komplett zeitzonen-naiv oder durchgehend in einer Standard-Zeitzone (z.B. UTC) vorliegen.
4. **Robustheit:** Der Parser muss unlesbare Dateien (z.B. Bilder, korrupte TXTs) im Ordner durch sauberes Error-Handling ignorieren können, ohne abzustürzen.
Benutze dafür die Tests in `/Users/lars/Teco/agenticTSParser/engine/validation.py`
## 4. Finaler Export (`train.csv`, `test.csv` & `croissant.json`)
Erst wenn das DataFrame komplett aufgebaut und validiert ist, führe **ganz am Ende** folgende Schritte aus:
- **Splitting:** Teile das finale DataFrame sinnvoll auf (z. B. 80% Train, 20% Test). Verhindere Data Leakage. Nutze keinen zufälligen Split, sondern trenne strikt chronologisch oder nach zusammenhängenden Blöcken der `target_col`.
- Speichere die Splits als `<dataset_dir>/_outputs/train.csv` und `<dataset_dir>/_outputs/test.csv`.
- **ML Croissant:** Generiere abschließend eine standardkonforme `croissant.json` (JSON-LD Format). Diese muss die Struktur der exportierten CSVs exakt beschreiben und alle generierten Features samt Datentypen (`sc:Float`, `sc:Integer`) auflisten.
- Regeln für die ML Croissant Metadaten (croissant.json):
Nutze nicht das Internet, um das Schema zu suchen. Halte dich exakt an diese Croissant 1.1 JSON-LD Strukturvorgaben:
Header (@context & @type): Das JSON muss mit dem Standard-Kontext für Schema.org und MLCommons beginnen. Der Haupttyp des Dokuments ist zwingend "@type": "sc:Dataset". Definiere grundlegende Felder wie name und description.
Dateien definieren (distribution): Erstelle eine Liste, die deine erzeugten CSVs auflistet. Jede Datei ist ein "@type": "cr:FileObject" und braucht eine @id (z.B. "train.csv"), einen name und den Pfad im contentUrl.
Tabellen definieren (recordSet): Erstelle für jeden Split (Train/Test) ein "@type": "cr:RecordSet".
Spalten mappen (field): Innerhalb jedes recordSet musst du jede Spalte des DataFrames als "@type": "cr:Field" definieren.
Das Wichtigste (Die Verknüpfung): Jedes Feld braucht seinen Datentyp (z.B. "sc:Float", "sc:DateTime", "sc:Integer") UND zwingend einen "source"-Block. Dieser Block erklärt, wo die Daten liegen. Er muss referenzieren: "fileObject": {"@id": "<Name_des_FileObjects>"} und "extract": {"column": "<Spaltenname>"}

## 5. Abschluss
- Berichte abschließend über das finale Alignment, die gewählte Splitting-Strategie (Regression vs. Classification), die Testergebnisse und den erfolgreichen Export der ML-Daten.