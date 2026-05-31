"""Programmatically build the three presentation notebooks with nbformat.

Creates:
  notebooks/01_eda.ipynb
  notebooks/02_summarization.ipynb
  notebooks/03_modeling.ipynb

Run:  python src/build_notebooks.py   (then execute with nbconvert)
"""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(x): return nbf.v4.new_markdown_cell(x)
def code(x): return nbf.v4.new_code_cell(x)


# Common setup cell injected at top of every notebook
SETUP = """\
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 120)
import config as C
DF = pd.read_parquet(C.SUBSET_PARQUET)
print("Loaded subset:", DF.shape)
DF.head(2)
"""


# ===========================================================================
# 01 - EDA
# ===========================================================================
def build_eda():
    cells = [
        md("# 01 · Exploratory Data Analysis — Multi-LexSum (Option 2)\n\n"
           "Civil-rights legal-case summarisation & classification.\n\n"
           "This notebook loads the **representative subset** built by "
           "`src/data_prep.py` (balanced across the most common case types, full "
           "case text recovered from the 2.2 GB `sources.json`) and reports the "
           "key findings about the data."),
        code(SETUP),
        md("## 1. What is in each column?\n"
           "Every row is one civil-rights case: the full (cleaned) text of all "
           "its court documents, three expert reference summaries, and metadata "
           "including the two modelling targets `class_action_sought` and "
           "`case_type`."),
        code('DF.dtypes\n'),
        code('DF[["case_id","case_name","case_type","class_action_sought",'
             '"n_documents","n_clean_chars","split"]].describe(include="all").T'),
        md("## 2. Split sizes & target distributions"),
        code('print("Split sizes:\\n", DF.split.value_counts(), "\\n")\n'
             'print("class_action_sought:\\n", DF.class_action_sought.value_counts())'),
        code('fig, ax = plt.subplots(1, 2, figsize=(13,4))\n'
             'DF.class_action_sought.value_counts().plot.bar(ax=ax[0], color="#4C72B0")\n'
             'ax[0].set_title("class_action_sought"); ax[0].tick_params(axis="x", rotation=0)\n'
             'DF.case_type.value_counts().plot.barh(ax=ax[1], color="#55A868")\n'
             'ax[1].set_title("case_type"); plt.tight_layout(); plt.show()'),
        md("## 3. Document length\n"
           "Legal cases are long — even after capping each case at 120k chars the "
           "median case is ~100k characters spread over several documents."),
        code('fig, ax = plt.subplots(1, 2, figsize=(13,4))\n'
             'DF.n_clean_chars.plot.hist(bins=40, ax=ax[0], color="#C44E52")\n'
             'ax[0].set_title("Cleaned characters per case"); ax[0].set_xlabel("chars")\n'
             'DF.n_documents.plot.hist(bins=30, ax=ax[1], color="#8172B3")\n'
             'ax[1].set_title("Documents per case"); ax[1].set_xlabel("# docs")\n'
             'plt.tight_layout(); plt.show()\n'
             'DF[["n_clean_chars","n_documents"]].describe().T'),
        md("## 4. Reference summary coverage & length\n"
           "All cases have a *long* summary; ~87% have a *short* one and ~47% a "
           "*tiny* one. Summary length shrinks sharply across the three tiers — "
           "this is the granularity our summariser must reproduce."),
        code('for col in ["summary_long","summary_short","summary_tiny"]:\n'
             '    s = DF[col].fillna("")\n'
             '    wl = s[s.str.len()>0].str.split().apply(len)\n'
             '    print(f"{col:14s} present={ (s.str.len()>0).sum():4d}  '
             'median_words={wl.median():.0f}  mean_words={wl.mean():.0f}")'),
        code('summ_words = pd.DataFrame({\n'
             '    t.split("_")[1]: DF[t].fillna("").str.split().apply(len).replace(0, np.nan)\n'
             '    for t in ["summary_long","summary_short","summary_tiny"]})\n'
             'summ_words.plot.box(figsize=(8,4)); plt.ylabel("words"); '
             'plt.title("Reference summary length by tier"); plt.show()'),
        md("## 5. Cross-tab: case type vs class action\n"
           "Some case types (e.g. Equal Employment, Prison/Jail Conditions) are "
           "much more likely to be brought as class actions — useful signal."),
        code('ct = pd.crosstab(DF.case_type, DF.class_action_sought, normalize="index")\n'
             'ct = ct.sort_values("Yes", ascending=False)\n'
             'ct.plot.barh(stacked=True, figsize=(9,5), color=["#4C72B0","#DD8452","#999"])\n'
             'plt.title("Share class-action-sought by case type"); plt.xlabel("share"); '
             'plt.legend(title="sought"); plt.tight_layout(); plt.show()\n'
             'ct.round(2)'),
        md("## 6. A peek at the raw vs cleaned text\n"
           "The raw documents carry court page-headers, form-feeds, OCR noise and "
           "broken unicode. `src/clean.py` strips the structural boilerplate and "
           "repairs the unicode (OCR typos are inherent to the scanned source)."),
        code('from clean import clean_case_text\n'
             'row = DF.iloc[0]\n'
             'print("RAW  :", row.full_text_raw[:500].replace(chr(10)," "))\n'
             'print("\\nCLEAN:", row.full_text_clean[:500].replace(chr(10)," "))'),
        md("### Key findings\n"
           "* 1,915 cases, 12 case types, official train/dev/test preserved.\n"
           "* Binary target `class_action_sought` is ~38% *Yes* (imbalanced).\n"
           "* `case_type` is multi-class; Equal Employment is the largest type.\n"
           "* Documents are very long (median ~100k chars) and noisy → cleaning + "
           "extractive summarisation are essential.\n"
           "* Reference summaries form a clear long ≫ short ≫ tiny length hierarchy."),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3"}
    nbf.write(nb, NB_DIR / "01_eda.ipynb")
    print("wrote 01_eda.ipynb")


# ===========================================================================
# 02 - Summarization
# ===========================================================================
def build_summarization():
    cells = [
        md("# 02 · Summarisation — long / short / tiny\n\n"
           "We build an **extractive** summariser (LexRank / TextRank / LSA over "
           "TF-IDF sentence graphs) that reproduces the dataset's three "
           "granularities, then compare our summaries to the expert reference "
           "summaries with **ROUGE**."),
        code(SETUP),
        md("## 1. The summariser\n"
           "`src/summarize.py` cleans the case, splits it into sentences and ranks "
           "them by graph centrality, returning the top-N for each tier "
           "(long≈12, short≈4, tiny=1 sentences)."),
        code('from summarize import three_tier_summary\n'
             'case = DF[DF.summary_tiny.fillna("").str.len()>0].iloc[3]\n'
             'print("CASE:", case.case_id, "|", case.case_name, "|", case.case_type)\n'
             'sysum = three_tier_summary(case.full_text_raw, method="lexrank")\n'
             'print("\\n--- OUR TINY ---\\n", sysum.tiny)\n'
             'print("\\n--- OUR SHORT ---\\n", sysum.short)'),
        md("## 2. Side-by-side with the expert reference"),
        code('print("OUR  TINY :", sysum.tiny)\n'
             'print("REF  TINY :", case.summary_tiny)\n'
             'print()\n'
             'print("OUR  SHORT:", sysum.short[:600])\n'
             'print("\\nREF  SHORT:", (case.summary_short or "")[:600])'),
        md("## 3. ROUGE evaluation against reference summaries\n"
           "ROUGE measures n-gram / longest-common-subsequence overlap between our "
           "extractive summaries and the expert ones. Extractive summaries of very "
           "long source docs won't match word-for-word, but ROUGE-1/2/L still show "
           "how much salient content we recover. We score on a sample for speed."),
        code('from rouge_score import rouge_scorer\n'
             'from summarize import summarize\n'
             'scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)\n'
             '\n'
             'def eval_tier(df, ref_col, n_sent, sample=40):\n'
             '    sub = df[df[ref_col].fillna("").str.len()>0].head(sample)\n'
             '    rows = []\n'
             '    for r in sub.itertuples():\n'
             '        hyp = summarize(r.full_text_raw, n_sent, method="lexrank")\n'
             '        sc = scorer.score(getattr(r, ref_col), hyp)\n'
             '        rows.append({k: sc[k].fmeasure for k in sc})\n'
             '    return pd.DataFrame(rows).mean()\n'
             '\n'
             'res = pd.DataFrame({\n'
             '    "tiny (1)":  eval_tier(DF, "summary_tiny", 1),\n'
             '    "short (4)": eval_tier(DF, "summary_short", 4),\n'
             '    "long (12)": eval_tier(DF, "summary_long", 12),\n'
             '})\n'
             'print(res.round(3))'),
        code('res.T.plot.bar(figsize=(8,4)); plt.title("ROUGE F1 vs expert summaries '
             'by tier"); plt.ylabel("F1"); plt.xticks(rotation=0); '
             'plt.legend(loc="upper right"); plt.tight_layout(); plt.show()'),
        md("## 4. Comparing summariser algorithms (LexRank vs TextRank vs LSA)"),
        code('def eval_method(method, sample=30):\n'
             '    sub = DF[DF.summary_short.fillna("").str.len()>0].head(sample)\n'
             '    rows=[]\n'
             '    for r in sub.itertuples():\n'
             '        hyp = summarize(r.full_text_raw, 4, method=method)\n'
             '        sc = scorer.score(r.summary_short, hyp)\n'
             '        rows.append(sc["rougeL"].fmeasure)\n'
             '    return np.mean(rows)\n'
             'pd.Series({m: eval_method(m) for m in ["lexrank","textrank","lsa"]},\n'
             '          name="rougeL-F1 (short)").round(3)'),
        md("### Findings\n"
           "* Our extractive summaries align with the expert ones on the main "
           "actors/claims; ROUGE is highest for the *long* tier (more content to "
           "overlap) and lowest for *tiny* (one sentence must hit the gist).\n"
           "* Extractive methods can only reuse source sentences, so they cannot "
           "match the abstractive paraphrasing of the experts — a noted limitation.\n"
           "* LexRank is the most robust on these long, noisy documents."),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3"}
    nbf.write(nb, NB_DIR / "02_summarization.ipynb")
    print("wrote 02_summarization.ipynb")


# ===========================================================================
# 03 - Modeling
# ===========================================================================
def build_modeling():
    cells = [
        md("# 03 · Modelling — class_action_sought & case_type\n\n"
           "Two classification tasks on the cleaned case text (TF-IDF features):\n\n"
           "1. **`class_action_sought`** — binary (Yes / No)\n"
           "2. **`case_type`** — multi-class (12 civil-rights types)\n\n"
           "Models: **Naive Bayes**, **Logistic Regression (GLM)**, **Linear SVM** "
           "and a **TensorFlow Bi-LSTM**. We train on train+dev and report on the "
           "held-out test split. (Full training incl. the LSTM is in "
           "`src/train_models.py`; here we retrain the fast linear models live and "
           "load the saved LSTM history.)"),
        code(SETUP),
        code('from modeling import make_pipeline, evaluate, top_features_per_class\n'
             'metrics = json.load(open(C.MODELS_DIR/"metrics.json"))\n'
             'top_feats = json.load(open(C.MODELS_DIR/"top_features.json"))\n'
             'DF["text"] = DF.full_text_clean.fillna("")\n'
             '\n'
             'def split_xy(target):\n'
             '    d = DF.copy()\n'
             '    if target=="class_action_sought": d=d[d[target].isin(["Yes","No"])]\n'
             '    d = d[d[target].astype(str).str.len()>0]\n'
             '    tr=d[d.split.isin(["train","dev"])]; te=d[d.split=="test"]\n'
             '    return (tr.text.tolist(), tr[target].astype(str).tolist(),\n'
             '            te.text.tolist(), te[target].astype(str).tolist())'),
        md("## 1. Train the linear models on both targets"),
        code('fitted = {}\n'
             'for target in ["class_action_sought","case_type"]:\n'
             '    Xtr,ytr,Xte,yte = split_xy(target)\n'
             '    labels = sorted(set(ytr)|set(yte))\n'
             '    fitted[target] = {"data":(Xte,yte,labels), "pipes":{}}\n'
             '    print(f"\\n### {target}  (train={len(Xtr)} test={len(Xte)} classes={len(labels)})")\n'
             '    for key in ["naive_bayes","logreg","linear_svm"]:\n'
             '        pipe = make_pipeline(key).fit(Xtr,ytr)\n'
             '        res = evaluate(key, yte, pipe.predict(Xte), labels=labels)\n'
             '        fitted[target]["pipes"][key]=pipe\n'
             '        print("   "+res.summary_line())'),
        md("## 2. Overall accuracy leaderboard (incl. the LSTM)\n"
           "Linear models on TF-IDF beat the LSTM here: the documents are long and "
           "the subset is modest, so sparse bag-of-words features generalise better "
           "than a sequence model trained from scratch."),
        code('rows=[]\n'
             'for target in metrics:\n'
             '    for m,d in metrics[target]["models"].items():\n'
             '        rows.append({"target":target,"model":m,"accuracy":d["accuracy"],"macro_f1":d["macro_f1"]})\n'
             'lead = pd.DataFrame(rows)\n'
             'fig,ax=plt.subplots(1,2,figsize=(13,4))\n'
             'for i,t in enumerate(metrics):\n'
             '    sub=lead[lead.target==t].sort_values("accuracy")\n'
             '    ax[i].barh(sub.model, sub.accuracy, color="#4C72B0")\n'
             '    ax[i].set_title(t); ax[i].set_xlim(0,1)\n'
             '    for y,v in enumerate(sub.accuracy): ax[i].text(v+.01,y,f"{v:.3f}",va="center")\n'
             'plt.tight_layout(); plt.show()\n'
             'lead'),
        md("## 3. Accuracy by genre/class (per-class accuracy)\n"
           "The project asks for accuracy **by class**, not just overall. "
           "Below: per-class accuracy (recall) of the best model on each target."),
        code('best = {"class_action_sought":"linear_svm","case_type":"linear_svm"}\n'
             'for target,m in best.items():\n'
             '    pca = metrics[target]["models"][m]["per_class_accuracy"]\n'
             '    s = pd.Series(pca).sort_values()\n'
             '    s.plot.barh(figsize=(8, 0.4*len(s)+1), color="#55A868")\n'
             '    plt.title(f"Per-class accuracy — {target} ({m})"); plt.xlim(0,1)\n'
             '    plt.tight_layout(); plt.show()'),
        md("## 4. Confusion matrix (case_type, Linear SVM)"),
        code('import numpy as np\n'
             'cm = np.array(metrics["case_type"]["models"]["linear_svm"]["confusion"])\n'
             'labs = metrics["case_type"]["labels"]\n'
             'cmn = cm/cm.sum(axis=1,keepdims=True)\n'
             'fig,ax=plt.subplots(figsize=(9,8))\n'
             'im=ax.imshow(cmn,cmap="Blues"); plt.colorbar(im,fraction=0.046)\n'
             'ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs,rotation=90,fontsize=8)\n'
             'ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=8)\n'
             'ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("case_type confusion (normalised)")\n'
             'plt.tight_layout(); plt.show()'),
        md("## 5. Most important words per class + word clouds\n"
           "TF-IDF + linear models are interpretable: the highest-weighted features "
           "per class read like a glossary of each civil-rights area."),
        code('for cls, words in list(top_feats["case_type"]["logreg"].items())[:6]:\n'
             '    print(f"{cls:38s}: {\", \".join(words[:10])}")'),
        code('try:\n'
             '    from wordcloud import WordCloud\n'
             '    classes = ["Yes","No"]\n'
             '    fig,ax=plt.subplots(1,2,figsize=(12,4))\n'
             '    for i,cls in enumerate(classes):\n'
             '        words = top_feats["class_action_sought"]["logreg"][cls]\n'
             '        freqs = {w: len(words)-j for j,w in enumerate(words)}\n'
             '        wc = WordCloud(width=500,height=300,background_color="white").generate_from_frequencies(freqs)\n'
             '        ax[i].imshow(wc); ax[i].axis("off"); ax[i].set_title(f"class_action_sought = {cls}")\n'
             '    plt.tight_layout(); plt.show()\n'
             'except Exception as e:\n'
             '    print("wordcloud unavailable:", e)'),
        code('try:\n'
             '    from wordcloud import WordCloud\n'
             '    show = ["Equal Employment","Prison Conditions","Immigration and/or the Border","Policing"]\n'
             '    fig,ax=plt.subplots(2,2,figsize=(12,7)); ax=ax.ravel()\n'
             '    for i,cls in enumerate(show):\n'
             '        words = top_feats["case_type"]["logreg"].get(cls,[])\n'
             '        freqs = {w: len(words)-j for j,w in enumerate(words)}\n'
             '        wc = WordCloud(width=500,height=300,background_color="white").generate_from_frequencies(freqs)\n'
             '        ax[i].imshow(wc); ax[i].axis("off"); ax[i].set_title(cls)\n'
             '    plt.tight_layout(); plt.show()\n'
             'except Exception as e: print("wordcloud unavailable:", e)'),
        md("## 6. The TensorFlow model: architecture & accuracy over epochs"),
        code('from modeling import build_lstm\n'
             'lstm = build_lstm(vocab_size=20000, maxlen=400, n_classes=12)\n'
             'lstm.summary()'),
        code('# Optional rendered diagram (needs graphviz+pydot). Falls back to the\n'
             '# text summary above if those system deps are unavailable.\n'
             'arch_png = C.MODELS_DIR / "lstm_arch.png"\n'
             'try:\n'
             '    from tensorflow.keras.utils import plot_model\n'
             '    plot_model(lstm, to_file=str(arch_png), show_shapes=True)\n'
             'except Exception as e:\n'
             '    print("plot_model needs graphviz/pydot; using the text summary above.")\n'
             'if arch_png.exists():\n'
             '    from IPython.display import Image, display; display(Image(str(arch_png)))'),
        code('fig,ax=plt.subplots(1,2,figsize=(13,4))\n'
             'for i,target in enumerate(["class_action_sought","case_type"]):\n'
             '    h = metrics[target]["models"]["lstm"]["history"]\n'
             '    ax[i].plot(h["accuracy"], marker="o", label="train")\n'
             '    ax[i].plot(h["val_accuracy"], marker="s", label="val")\n'
             '    ax[i].set_title(f"LSTM accuracy/epoch — {target}"); ax[i].set_xlabel("epoch")\n'
             '    ax[i].set_ylabel("accuracy"); ax[i].legend()\n'
             'plt.tight_layout(); plt.show()'),
        md("## 7. Where do the models disagree?\n"
           "We compare Naive Bayes vs Linear SVM on `case_type` and inspect cases "
           "they label differently. Disagreements cluster on **overlapping civil-"
           "rights areas** (e.g. *Prison Conditions* vs *Jail Conditions*, or "
           "*Policing* vs *Criminal Justice*): NB weights individual indicative "
           "words and is swayed by frequent generic legal terms, while the "
           "margin-based SVM uses the whole weighted bag-of-words, so it separates "
           "the near-duplicate categories better."),
        code('Xte,yte,labels = fitted["case_type"]["data"]\n'
             'nb_pred = fitted["case_type"]["pipes"]["naive_bayes"].predict(Xte)\n'
             'svm_pred = fitted["case_type"]["pipes"]["linear_svm"].predict(Xte)\n'
             'dis = pd.DataFrame({"true":yte,"naive_bayes":nb_pred,"linear_svm":svm_pred})\n'
             'dis["snippet"]=[t[:160].replace(chr(10)," ") for t in Xte]\n'
             'mask = dis.naive_bayes!=dis.linear_svm\n'
             'print(f"{mask.sum()} / {len(dis)} test cases disagree between NB and SVM")\n'
             'dis[mask].head(8)[["true","naive_bayes","linear_svm","snippet"]]'),
        code('# which pairs of classes get confused between the two models\n'
             'pairs = (dis[mask].groupby(["naive_bayes","linear_svm"]).size()\n'
             '         .sort_values(ascending=False).head(8))\n'
             'pairs'),
        md("### Findings\n"
           "* **Linear SVM** is the best model on both targets "
           "(class_action ≈ 0.95, case_type ≈ 0.91).\n"
           "* The **GLM (logistic regression)** is a close second and the most "
           "calibrated → it is the default in the interactive tool.\n"
           "* **Naive Bayes** trails because its independence assumption over "
           "long legal text over-counts generic terms.\n"
           "* The **LSTM** underperforms the linear models on this subset size — "
           "long documents truncated to 400 tokens lose signal, and 1.5k examples "
           "is little for a sequence model trained from scratch.\n"
           "* Per-class accuracy is high for distinctive types (Immigration, Equal "
           "Employment) and lowest for overlapping ones (Jail vs Prison Conditions)."),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3"}
    nbf.write(nb, NB_DIR / "03_modeling.ipynb")
    print("wrote 03_modeling.ipynb")


if __name__ == "__main__":
    build_eda()
    build_summarization()
    build_modeling()
