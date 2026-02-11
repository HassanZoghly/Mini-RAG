from fastapi import FastAPI
from dotenv import load_dotenv # Run before routes to ensure routes will see it
load_dotenv(".env")  # Load environment variables from .env file

from routes import base
app = FastAPI()

app.include_router(base.base_router)
