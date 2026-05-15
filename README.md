# Mini RAG

This is a minimal implemetation of a RAG Model for question answering.

## What to know?

- .env.example ==> environment variables example file for env.
- assets ==> folder containing images, icons etc that will help.
- helpers.config ==> Contains the logic to load environment variables .env.
- any function dealing with database is async when call it must do "await" before it.

## Requirements

- Python 3.8+
- Conda
1. Create a new environment using the following command:
```bash
conda create -n mini-rag python=3.8
```
2. Activate the environment:
```bash
conda activate mini-rag
```

## Installation

### Install the required packages
```bash
pip install -r requirements.txt
```

### System Dependencies
If you want to use the OCR features for Images and Scanned PDFs, you must install Tesseract OCR on your system.
* Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
* MacOS: `brew install tesseract`

### Setup the environment variables
```bash
cp .env.example .env
```

## Run Docker Compose Services

```bash
$ cd docker
$ cp .env.example .env
```

- update `.env` with your credentials

```bash
$ cd docker
$ sudo docker compose up -d
```

##### Just to check if there are services running
```bash
$ sudo docker stop $(sudo docker ps -aq) # stop all running containers
$ sudo docker rm $(sudo docker ps -aq) # remove all containers
$ sudo docker rmi $(sudo docker images -q) # remove all images
$ sudo docker volume $(sudo docker volume ls -q) # remove all volumes
$ sudo docker system prune --all # remove all unused containers, networks, images, and optionally, volumes.
```

## Running the Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```
- Access the application at `http://localhost:5000`.
- host give access from other devices in the network.
