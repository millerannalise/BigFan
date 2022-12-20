import sys
from os import path

from setuptools import setup

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, "BigFan", "_version.py")) as f:
    __version__ = ""
    exec(f.read())

with open(path.join(HERE, "README.md")) as f:
    readme = f.read()

with open(path.join(HERE, "CHANGELOG.md")) as f:
    changelog = f.read()


desc = readme + "\n\n" + changelog
try:
    import pypandoc

    long_description = pypandoc.convert_text(desc, "rst", format="md")
    with open(path.join(HERE, "README.rst"), "w") as rst_readme:
        rst_readme.write(long_description)
except (ImportError, OSError, IOError):
    long_description = desc


def requirements_from_txt(fname: str):
    txt_path = path.join(HERE, fname)
    with open(txt_path, "r") as txt_file:
        dependencies = txt_file.readlines()
        dependencies = [d.strip() for d in dependencies]
    return dependencies


install_requires = requirements_from_txt("requirements.txt")
tests_require = requirements_from_txt("requirements_test.txt")

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
setup_requires = ["pytest-runner"] if needs_pytest else []

setup(
    name="BigFan",
    version=__version__,
    python_requires=">=3.7.*",
    packages=["BigFan"],
    url="https://github.com/millerannalise/BigFan",
    license="MIT",
    author="Annalise Miller",
    author_email="annalise.mckenzie.miller@gmail.com",
    package_dir={"BigFan": "BigFan"},
    package_data={"BigFan": [path.join("lookup_data", "*")],},
    description="Wind Resource Assessment Tools",
    long_description=long_description,
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    py_modules=["six"],
)
