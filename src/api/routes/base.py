from fastapi import APIRouter, FastAPI, Depends
import os
from core.settings import get_settings, Settings

base_router = APIRouter(
    prefix="/v1" # All routes in this router will be prefixed with /v1
)

@base_router.get("/")
async def welcome(app_settings: Settings = Depends(get_settings)):

    app_name = app_settings.APP_NAME

    return {
        "message": f"Welcome to the My << {app_name} >> FastAPI application!"
    }
