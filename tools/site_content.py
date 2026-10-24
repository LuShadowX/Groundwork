"""Text and figures for the Groundwork site (used by build_site.py)."""

NAME = "Groundwork"
REPO = "https://github.com/LuShadowX/Groundwork"
ACCENT = "#2a7f62"
SOFT = "#e8f3ee"
TAGLINE = "Foundations of machine learning, worked through on real Kaggle data: NumPy, pandas, EDA, the maths, optimisers from scratch and learning theory."
HEADLINE = "The foundations of machine learning, on one real dataset."
INTRO = [
    "Six notebooks, one per unit of CSE273. Every idea is computed on the Kaggle Playground S6E4 "
    "<i>Predicting Irrigation Need</i> data (630,000 fields), and every formula is checked against a library "
    "or a numerical answer: the bias–variance split is measured, gradients are verified with finite differences, "
    "gradient descent, Momentum, RMSProp and Adam are written from scratch, and PAC bounds are tested empirically.",
    "The reusable parts live in a small tested package, <code>groundwork</code>; the notebooks run unchanged on Kaggle.",
]
STATS = [("6", "notebooks, one per unit"), ("57", "syllabus topics mapped"), ("4", "optimisers written from scratch"), ("16", "unit tests")]
FIGURES = [
    ("g01-bias-variance", "Bias² and variance measured with 30 bootstrap fits per tree depth."),
    ("g03-heatmap", "Correlation of every numeric feature with irrigation need."),
    ("g04-bayes", "Bayes' theorem, one piece of evidence at a time."),
    ("g05-rosenbrock", "Gradient descent, Momentum, RMSProp and Adam (from scratch) on the Rosenbrock valley."),
    ("g05-ci-coverage", "500 confidence intervals: about 95% contain the true mean."),
    ("g06-erm", "Empirical vs true risk under ERM, as the sample grows."),
]
RUN = """git clone https://github.com/LuShadowX/Groundwork && cd Groundwork
pip install -e ".[dev]"
kaggle competitions download -c playground-series-s6e4 -p data && unzip data/*.zip -d data
pytest                      # 16 tests
jupyter lab notebooks/      # or open any notebook on Kaggle with the competition attached"""
FOOTER = "Lukshya Singh Supyal · B.Tech CSE · built for CSE273 Foundations of Machine Learning · data: Kaggle Playground Series S6E4"
OUTCOME_TITLE = "Course outcomes"
