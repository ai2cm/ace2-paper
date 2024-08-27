# manually pip install more recent beaker-py, since conda-forge channel
# is missing latest versions.
create_environment:
	conda env create -f environment.yaml
	conda run -n ace2-paper pip install beaker-py==1.30