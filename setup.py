from setuptools import setup
import setuptools

setup(
    name='cntext',     # Package name (keeping for compatibility; consider 'entext' for future releases)
    version='3.0.0',   # Major version bump for English adaptation fork
    description='English text analysis library forked from cntext, specialized for social science research with word embeddings, sentiment analysis, semantic projection, and readability metrics',
    author='Fork Maintainer',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    url='https://github.com/yourusername/cntext',  # Replace with your fork URL
    packages=setuptools.find_packages(),
    package_data = {'':['dict/*.yaml', 'font/*.ttf']},
    include_package_data=True,
    install_requires=['jieba>=0.42', 'scikit-learn==1.5.0', 'numpy==1.26.4', 'matplotlib',
                      'gensim==4.3.2', 'nltk', 'pandas', 'chardet', 'h5py', 'networkx', 'distinctiveness',
                      'tqdm', 'python-docx', 'scipy>=1.12.0', 'scienceplots', 'PyMuPDF', 'ftfy', 'opencc-python-reimplemented',
                      "ollama>=0.2.1", "pydantic>=2.8.2", "instructor>=1.6.0", "openai>=1.61.1","contractions>=0.1.73"],
    extras_require={
        'english': ['spacy>=3.0.0'],  # Optional: Enhanced English NLP support
    },
    python_requires='>=3.9',  # Updated to match documentation
    license="MIT",
    keywords=['english', 'text mining', 'sentiment', 'sentiment analysis', 'natural language processing',
              'semantic projection', 'text similarity', 'GloVe', 'word2vec', 'social science',
              'readability', 'word embeddings', 'semantic analysis'],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown")
    #py_modules = ['eventextraction.py']
