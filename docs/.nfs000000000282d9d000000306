#!/bin/sh
git checkout gh-pages
cd ../
git checkout master docs
cd docs
ln -sf ../nose/testdata
ln -sf ../../../nose/testdata tutorials/example_code
make clean
make html
cd ../
rm -rf _images _modules _sources _static plot_directive tutorials docs/testdata docs/tutorials/example_code/testdata
mv -fv docs/_build/html/* ./
git add -A
git commit -m 'docs update'
git push origin gh-pages
git checkout master



