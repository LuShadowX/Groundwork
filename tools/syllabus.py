"""CSE273 Foundations of Machine Learning: every syllabus topic and where it is covered.

Used by build_site.py for the course map. Each entry is (topic, notebook, section).
"""

COURSE = "CSE273 · Foundations of Machine Learning"

NOTEBOOKS = {
    "01_foundations_numpy": "Machine learning foundations and NumPy",
    "02_pandas_matplotlib": "Pandas for data handling",
    "03_eda": "Exploratory data analysis",
    "04_linear_algebra_probability_calculus": "Linear algebra, probability and calculus",
    "05_statistics_optimisation": "Statistics for ML and optimisation",
    "06_learning_theory": "Statistical learning theory",
}

UNITS = [
    ("I", "Machine Learning Foundation", "01_foundations_numpy", [
        ("Introduction to machine learning", "1"),
        ("AI vs machine learning vs data science", "2"),
        ("Supervised, unsupervised and reinforcement learning", "3"),
        ("Parametric vs non-parametric models", "4"),
        ("Generative vs discriminative models", "4"),
        ("Machine learning workflow", "5"),
        ("Train–test split, overview of evaluation metrics", "6"),
        ("Overfitting and underfitting", "6"),
        ("Bias–variance trade-off (measured with 30 bootstrap fits)", "6"),
        ("Applications and use cases", "7"),
        ("NumPy: why NumPy, creating arrays, operations", "8"),
        ("Broadcasting rule, matrix operations, random module", "8"),
    ]),
    ("II", "Pandas for Data Handling", "02_pandas_matplotlib", [
        ("Pandas Series and DataFrame", "1"),
        ("Reading and writing CSV files", "2"),
        ("Inspecting datasets", "3"),
        ("Selection: loc vs iloc; boolean filtering", "4"),
        ("Line, bar, histogram, scatter and pie charts", "5"),
        ("Customising plots", "6"),
        ("Subplots", "7"),
    ]),
    ("III", "Exploratory Data Analysis", "03_eda", [
        ("Types of data (quantitative, qualitative)", "1"),
        ("Loading data and initial inspection", "2"),
        ("Univariate analysis: histogram, box plot, count plot", "3"),
        ("Bivariate analysis: scatter, line, violin plots", "4"),
        ("Correlation heatmaps and multicollinearity (VIF)", "5"),
        ("Skewness and kurtosis", "6"),
        ("Patterns, anomalies (outliers) and trends", "7"),
        ("EDA workflow integration", "8"),
    ]),
    ("IV", "Linear Algebra and Calculus", "04_linear_algebra_probability_calculus", [
        ("Data structures: scalar, vector, matrix, tensor", "1.1"),
        ("Eigenvalues and eigenvectors (power iteration from scratch)", "1.2"),
        ("Random variables and probability distributions", "2.1–2.2"),
        ("Statistical measures", "2.3"),
        ("Types of probability: joint, marginal, conditional", "2.4"),
        ("Bayes' theorem and inference", "3.1"),
        ("Likelihood vs probability", "3.2"),
        ("Functions, partial derivatives and the gradient", "4.1–4.3"),
        ("The chain rule", "4.4"),
    ]),
    ("V", "Statistics for ML", "05_statistics_optimisation", [
        ("Sampling and population", "1.1"),
        ("Hypothesis testing", "1.2"),
        ("Confidence intervals", "1.3"),
        ("Correlation", "1.4"),
        ("Maximum likelihood estimation", "2.1"),
        ("Loss (cost) functions", "2.2"),
        ("Convex vs non-convex functions", "3.1"),
        ("Minima and saddle points", "3.2"),
        ("Gradient descent and the learning rate", "4.1–4.2"),
        ("Variants of gradient descent (batch, stochastic, mini-batch)", "4.3"),
        ("Momentum, RMSProp and Adam (from scratch)", "5"),
    ]),
    ("VI", "Statistical Learning Theory", "06_learning_theory", [
        ("Well-posed learning problems (T-P-E)", "1"),
        ("Components of a learning system", "2"),
        ("Statistical learning framework", "3"),
        ("Empirical risk minimisation, overfitting and generalisation", "4"),
        ("Inductive bias", "5"),
        ("Role of prior knowledge", "6"),
        ("PAC learning", "7"),
        ("Sample complexity: finite hypothesis space and VC dimension", "8"),
        ("No Free Lunch theorem", "9"),
        ("Choosing algorithms; Occam's razor", "10"),
    ]),
]

# The course page lists the six units but no numbered course outcomes, so the map is by
# syllabus topic. (Official COs from UMS can be added here as (code, text, notebook) tuples.)
OUTCOMES = []
