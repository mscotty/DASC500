# Dockerfile
# Use the Jupyter PySpark Notebook image, which includes Jupyter Lab and the 'jovyan' user.
# Check Docker Hub for the latest tag with Spark 4.0.0:
# https://hub.docker.com/r/jupyter/pyspark-notebook/tags
FROM jupyter/pyspark-notebook:x86_64-spark-3.5.0

# Switch to the 'jovyan' user, which exists in this base image
USER jovyan

# --- Create and activate a dedicated virtual environment for the notebook's dependencies ---
ENV VIRTUAL_ENV=/home/jovyan/venv_notebook
RUN python3 -m venv $VIRTUAL_ENV

# Activate the virtual environment for all subsequent commands in this Dockerfile
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# --- Set the working directory for your application files within the container ---
# This will be the directory Jupyter Lab opens to by default
WORKDIR /home/jovyan/app

# --- Copy and install Python dependencies into the virtual environment ---
# Copy the requirements.txt file first for better Docker build caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Install ipykernel and register the virtual environment as a Jupyter kernel ---
# This allows you to select your custom environment inside Jupyter Lab
RUN pip install ipykernel
RUN python3 -m ipykernel install --user --name=my_pyspark_env --display-name "PySpark (my_pyspark_env)"

# --- Copy your entire local project directory into the container ---
# The '.' copies everything from your Docker build context (C:\Users\amber\Documents\Mitchell\school\2025\DASC500\DASC500)
# into the container's WORKDIR (/home/jovyan/app)
COPY . .

# The base 'jupyter/pyspark-notebook' image already sets the default command to start Jupyter Lab.
# You typically don't need to specify EXPOSE or CMD here unless you want to override
# the base image's default behavior (e.g., for different ports or starting a specific script).