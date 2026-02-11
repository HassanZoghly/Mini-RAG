# Mini RAG

This is a minimal implemetation of a RAG Model for question answering.

## What to know?

- .env.example ==> environment variables example file for env.
- assets ==> folder containing images, icons etc that will help.


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

### Setup the environment variables
```bash
cp .env.example .env
```

## Running the Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```
- Access the application at `http://localhost:5000`.
- host give access from other devices in the network.
