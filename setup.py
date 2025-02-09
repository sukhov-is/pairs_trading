from setuptools import setup, find_packages

setup(
    name="pair_trading_system",
    version="1.0.0",
    description="Система парного трейдинга",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "yfinance>=0.2.0",
        "pandas-datareader>=0.10.0",
        "statsmodels>=0.13.0",
        "optuna>=3.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=0.20.0",
        "joblib>=1.1.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'isort>=5.10.0',
            'mypy>=0.950',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    python_requires='>=3.8',
)