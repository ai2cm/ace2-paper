ENVIRONMENT_NAME=ace2-paper

create_environment:
	conda create --yes -n $(ENVIRONMENT_NAME) -c conda-forge python=3.10 pip
	conda run --no-capture-output -n $(ENVIRONMENT_NAME) pip install -r requirements.txt
