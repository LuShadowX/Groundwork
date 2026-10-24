# Groundwork

The foundations of machine learning, worked through on one real dataset. Built for **CSE273 Foundations of Machine Learning** (LPU).

**Site:** https://lushadowx.github.io/Groundwork/ (every notebook with its outputs, plus a course map)

All six units of the syllabus are covered, one notebook each, on the Kaggle [Playground Series S6E4: Predicting Irrigation Need](https://www.kaggle.com/competitions/playground-series-s6e4) data: 630,000 fields described by soil, weather and crop features, labelled Low / Medium / High irrigation need. I entered that competition in April 2026; this project goes back to the foundations under the model.

| Notebook | Unit | What's in it |
|---|---|---|
| [01 Foundations and NumPy](notebooks/01_foundations_numpy.ipynb) | I | ML types and workflow, parametric vs non-parametric, generative vs discriminative, overfitting curve, bias² and variance *measured*, NumPy (broadcasting, matrices, random) |
| [02 Pandas and Matplotlib](notebooks/02_pandas_matplotlib.ipynb) | II | Series/DataFrame, CSV, `loc`/`iloc`, filtering, group-by, every chart type, customisation, subplots |
| [03 EDA](notebooks/03_eda.ipynb) | III | univariate and bivariate plots, correlation and VIF, skewness and kurtosis, outlier rules tested on injected faults, patterns and trends |
| [04 Linear algebra, probability, calculus](notebooks/04_linear_algebra_probability_calculus.ipynb) | IV | eigenvectors (power iteration from scratch), distributions, joint/marginal/conditional probability, Bayes' theorem on the data, gradients checked numerically, the chain rule |
| [05 Statistics and optimisation](notebooks/05_statistics_optimisation.ipynb) | V | CLT, t/χ²/permutation tests, confidence intervals (incl. my `conviction` library), MLE, loss functions, convexity, saddle points, GD variants, **Momentum / RMSProp / Adam from scratch** |
| [06 Learning theory](notebooks/06_learning_theory.ipynb) | VI | ERM measured against true risk, overfitting with label noise, inductive bias, prior knowledge (monotonic constraints), **PAC bound tested empirically**, a certified error bound with `conviction.RiskGuard`, VC dimension by shattering, No Free Lunch by exhaustive enumeration, Occam's razor |

## The package

`src/groundwork/` holds the parts worth testing:

- `optim.py`: gradient descent, Momentum, RMSProp and Adam (one `step(w, g)` each), Rosenbrock and saddle test functions, softmax regression trained with any optimiser
- `stats.py`: numerical gradients, bootstrap CIs, normal MLE, loss functions
- `theory.py`: PAC and VC sample-size bounds, a linear-separability LP, shattering, ERM over decision stumps
- `data.py`, `plotting.py`: loading the competition data, one figure style

## Run it

```bash
git clone https://github.com/LuShadowX/Groundwork && cd Groundwork
pip install -e ".[dev]"
kaggle competitions download -c playground-series-s6e4 -p data && unzip data/playground-series-s6e4.zip -d data
pytest                       # 16 tests
jupyter lab notebooks/
python tools/build_site.py   # rebuild docs/ from the executed notebooks
```

On Kaggle, attach the competition to a notebook; the first cell installs the package from this repo.

The competition data isn't committed (Kaggle's rules don't allow redistributing it).

Lukshya Singh Supyal · B.Tech CSE
