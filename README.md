# tract-clustering
Data-driven approach for identifying white matter tracts in quantitative histology of the spinal cord. This 
code was applied to the rat spinal cord, but it could equally be applied to other species.

## Citation

Nami H, Perone CS and Cohen-Adad J (2022) _Histology-informed automatic parcellation of white matter tracts in the rat spinal cord_.
**Front. Neuroanat.** 16:960475. doi: [10.3389/fnana.2022.960475](https://www.frontiersin.org/articles/10.3389/fnana.2022.960475)

## Dependency

- Python 3.7 (conda distribution)
- [ANTs](http://stnava.github.io/ANTs/)

## Getting started

### Data

Download the [RatAtlas](https://osf.io/g7kx8/). For more information about how this dataset was generated, see [Saliani et al. Neuroimage 2019](https://www.ncbi.nlm.nih.gov/pubmed/31491525). 

### Code

Download this repository:
~~~
git clone https://github.com/neuropoly/tract-clustering.git
~~~

Install package and dependencies:
~~~
cd tract-clustering
pip install -e .
~~~

Edit the script `scripts/params.py` and update variables according to your files location

Run registration:
~~~
python register_slicewise.py
~~~

Slice-wise clustering:
~~~
python clustering_slicewise.py
~~~

Region-wise clustering:
~~~
# Run clustering
python clustering_regionwise.py
~~~
