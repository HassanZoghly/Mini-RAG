from fastapi import APIRouter, FastAPI
import os

base_router = APIRouter(
    prefix="/v1" # All routes in this router will be prefixed with /v1
)

@base_router.get("/")
async def welcome():
    app_name = os.getenv("APP_NAME")

    return {
        "message": f"Welcome to the My {app_name} FastAPI application!"
    }
