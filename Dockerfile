# Srat image from base python
FROM python:3.9

# Change working dir
WORKDIR /code

# Copy requirements
COPY  ./requirements.txt /code/requirements.txt

# Install FFMPEG
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy App
COPY ./App /code/App

# Copy other relevant stuff
COPY ./Utils /code/Utils
COPY ./Source /code/Source

# Chief among those models
COPY ./Models /code/Models

# Run the API
CMD [ "python", "-m", "App.app" ]
    # Jaja so many apps, yeah try doing it yourself