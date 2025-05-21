from setuptools import setup, find_packages

setup(
    name="shuwaemon-news-generator",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "jinja2>=3.1.2",
        "python-multipart>=0.0.6",
        "google-generativeai>=0.3.0",
        "pandas>=2.0.0",
        "openpyxl>=3.1.2",
    ],
    python_requires=">=3.8",
    author="Makoto World",
    author_email="your-email@example.com",
    description="しゅわえもんニュース生成システム",
    keywords="news, generation, AI, Google Gemini",
    url="https://github.com/yourusername/shuwaemon-news-generator",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
