# Srat image from base python
FROM python:3.9

# Change working dir
WORKDIR /code

# Copy requirements
COPY  ./requirements.txt /code/requirements.txt

# Install requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy App
COPY ./App /code/App

# Copy other relevant stuff
COPY ./Utils /code/Utils
COPY ./Source /code/Source

# Run the API
CMD [ "uvicorn", "App.app:app" ]
    # Jaja so many apps, yeah try doing it yourself