from setuptools import setup

setup(
    name='SelfNormalizingFlow',
    version='0.0.1',
    description="SelfNormalizingFlow",
    author="",
    author_email='',
    packages=[
        'snf'
    ],
    entry_points={
        'console_scripts': [
            'snf=snf.cli:main',
        ]
    },
    python_requires='>=3.6',
)