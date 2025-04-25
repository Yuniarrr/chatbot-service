from enum import Enum


class ERROR_MESSAGES(str, Enum):
    def __str__(self) -> str:
        return super().__str__()

    ACCESS_PROHIBITED = "Anda tidak memiliki izin untuk mengakses sumber daya ini. Silakan hubungi administrator Anda untuk mendapatkan bantuan."
    DUPLICATE_VALUE = (
        lambda name="": f"'{name}' sudah ada. Silakan gunakan {name} yang berbeda."
    )
    INVALID_TOKEN_OR_API_KEY = (
        lambda name="": f"Sesi Anda telah kedaluwarsa atau '{name}' tidak valid. Silakan masuk lagi."
    )
    MISSING_TOKEN_OR_API_KEY = lambda name="": f"Forbidden: '{name}' hilang"
    NOT_FOUND = lambda name="": f"Not Found: '{name}' tidak ditemukan"
    UNAUTHORIZED = "Un-Authorized: Tidak memiliki akses"
    EMPTY_CONTENT = "Empty content: content tidak ditemukan"
    FAILED_UPLOAD = "Gagal mengupload file"
    PANDOC_NOT_INSTALLED = (
        "Pandoc tidak terinstal. Silakan instal pandoc untuk menggunakan fitur ini."
    )


class SUCCESS_MESSAGE(str, Enum):
    def __str__(self) -> str:
        return super().__str__()

    CREATED = "Data berhasil dibuat"
    DELETED = "Data berhasil dihapus"
    RETRIEVED = "Data berhasil didapat"
