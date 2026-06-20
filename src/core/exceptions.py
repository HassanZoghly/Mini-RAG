from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional

class AppException(Exception):
    def __init__(self, status_code: int, message: str, details: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self.message = message
        self.details = details or {}

class ValidationException(AppException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(status_code=400, message=message, details=details)

class ProcessingException(AppException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(status_code=500, message=message, details=details)

async def app_exception_handler(request: Request, exc: AppException):
    content = {
        "error": True,
        "message": exc.message,
    }
    if exc.details:
        content.update(exc.details)

    return JSONResponse(
        status_code=exc.status_code,
        content=content
    )

async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal Server Error",
            "details": str(exc)
        }
    )
