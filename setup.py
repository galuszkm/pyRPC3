from setuptools import setup

setup(
    name="pyRPC3",
    version="1.0.0",
    package_dir={"pyRPC3": "src"},
    packages=["pyRPC3"],
    python_requires=">=3.11",
    install_requires=[
        "colorama==0.4.6",
        "contourpy==1.3.1",
        "cycler==0.12.1",
        "fonttools==4.56.0",
        "iniconfig==2.0.0",
        "kiwisolver==1.4.8",
        "matplotlib==3.10.1",
        "numpy==2.2.3",
        "packaging==24.2",
        "pillow==11.1.0",
        "pluggy==1.5.0",
        "pyparsing==3.2.1",
        "pytest==8.3.4",
        "python-dateutil==2.9.0.post0",
        "six==1.17.0",
    ],
)