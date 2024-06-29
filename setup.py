from setuptools import setup, find_packages

setup(
    name='ai_deception_detection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch',
        'numpy',
        'matplotlib',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for detecting deception using AI',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ai-deception-detection',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)