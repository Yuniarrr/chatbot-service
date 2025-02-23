from fastapi import HTTPException


class DuplicateValueException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=409, detail=detail)


class DatabaseException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=f"Database Error: {detail}")


class NotFoundException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=404, detail=detail)


class UnauthorizedException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=401, detail=detail)


class BadRequestException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)


class ForbiddenException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=403, detail=detail)


class InternalServerException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail)
