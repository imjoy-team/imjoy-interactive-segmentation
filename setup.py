"""Set up file for imjoy-interactive-trainer."""
import json
from pathlib import Path

from setuptools import setup, find_packages

PROJECT_DIR = Path(__file__).parent.resolve()
PACKAGE_JSON = (PROJECT_DIR / "imjoy_interactive_trainer" / "VERSION").read_text()

with open(PROJECT_DIR/ 'requirements.txt') as fp:
    install_requires = [req for req in fp.read().split('\n') if not req.startswith('#')]

print('====>', install_requires)
setup(
    name="imjoy-interactive-trainer",
    version=PACKAGE_JSON,
    install_requires=install_requires,
    description="Interactive segmentation and annotation with ImJoy",
    url="https://github.com/imjoy-team/imjoy-interactive-segmentation",
    author="imjoy-team",
    author_email="imjoy.team@gmail.com",
    license="ImJoy Team",
    packages=find_packages(include=["imjoy_interactive_trainer", "imjoy_interactive_trainer.*"]),
    include_package_data=True,
    zip_safe=False,
)