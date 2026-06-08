from setuptools import find_packages, setup


setup(
    name="prism-lmaas",
    version="0.1.0",
    description="PRISM workload-aware LMaaS management research prototype.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
)
