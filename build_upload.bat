del /Q .\dist\*
python setup.py sdist bdist_wheel
twine upload -u mmmaaaggg dist/*
