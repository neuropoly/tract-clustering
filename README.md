# tract-clustering
Data-driven approach for tracts in rat histology

## Dependency

- Python 3.7 (conda distribution)
- [ANTs](http://stnava.github.io/ANTs/)

## How to use

Data: 
- Download the [RatAtlas](https://osf.io/g7kx8/). For more information about how this dataset was generated, see [Saliani et al. Neuroimage 2019](https://www.ncbi.nlm.nih.gov/pubmed/31491525). 
- Unzip

Code:
- Download this code:
~~~
git clone https://github.com/neuropoly/tract-clustering.git
~~~

- Install dependencies
~~~
cd tract-clustering
pip install -e .
~~~

- Open the script `registration_script_clustering_edit.py`
- Update the variable `Folder` to point to the unzipped rat atlas folder.
 

