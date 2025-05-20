import json
import pandas as pd
import re
import sys
import asyncio

from typing import List, Dict
from langchain_core.documents import Document

from app.retrieval.vector_store import vector_store_service

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ganti 'nama_file.xlsx' dengan path ke file Excel Anda
xls = pd.ExcelFile("./parse/jadwal-mata-kuliah.xlsx")
df_jadwal = xls.parse("Jadwal Kuliah Genap 2425", header=None)

# Tampilkan beberapa baris awal untuk melihat strukturnya
df_jadwal.head(20)

# Mulai dari baris ke-6 sampai bawah (skip header deskriptif)
jadwal_data = df_jadwal.copy()

# Deteksi baris yang merupakan header kolom ruang
header_row_index = 7
sesi_col_index = 2
ruang_start_col_index = 3

# Ambil nama ruang dari header
ruang_headers = (
    jadwal_data.iloc[header_row_index, ruang_start_col_index:].fillna("").tolist()
)
ruang_headers = [
    re.sub(r"\n.*", "", str(r)).strip() for r in ruang_headers
]  # Hapus isi dalam kurung & newline

# Proses baris isi jadwal
jadwal_list: List[Dict] = []

current_day = None
for idx in range(header_row_index + 1, len(jadwal_data)):
    row = jadwal_data.iloc[idx]

    # Deteksi hari baru jika kolom sesi berisi 'Hari'
    if (
        str(row[sesi_col_index]).strip().lower() == "sesi"
        and "hari" in str(row[1]).lower()
    ):
        continue
    elif isinstance(row[1], str) and row[1].strip() != "":
        current_day = row[1].strip()

    jam = row[sesi_col_index]

    if pd.isna(jam) or pd.isna(current_day):
        continue

    for col_offset, cell in enumerate(row[ruang_start_col_index:], start=0):
        if pd.isna(cell) or not isinstance(cell, str):
            continue

        ruang = (
            ruang_headers[col_offset]
            if col_offset < len(ruang_headers)
            else f"Ruang-{col_offset}"
        )

        # Ekstraksi informasi dari isi cell
        mk_info = cell.strip().split("\n")
        mk_info_joined = " ".join(mk_info)
        mata_kuliah = mk_info_joined.split("Semester")[0].strip()
        semester = ""
        dosen = ""

        match = re.search(r"Semester\s+(.+?)\s*\|\s*(.+)", mk_info_joined)
        if match:
            semester = match.group(1).strip()
            dosen = match.group(2).strip()
        else:
            # Coba match yang hanya punya "Semester X"
            match_alt = re.search(r"Semester\s+(\d+)", mk_info_joined)
            if match_alt:
                semester = match_alt.group(1).strip()

        jadwal_list.append(
            {
                "hari": current_day,
                "jam": jam,
                "ruang": ruang,
                "mata_kuliah": mata_kuliah,
                "semester": semester,
                "dosen": dosen,
                "mk_info": mk_info,
            }
        )


with open("./parse/jadwal.json", "w", encoding="utf-8") as f:
    json.dump(jadwal_list, f, ensure_ascii=False, indent=4)

extra_meta = {
    "document_type": "jadwal",
    "topik": "jadwal mata kuliah bersama",
    "tahun_ajaran": "2024/2025",
    "file_id": "fb0b2e1f-66cb-418a-8215-3204ca02d085",
    "file_name": "fb0b2e1f-66cb-418a-8215-3204ca02d085_PUBLISH Mahasiswa - Perkuliahan Sem Genap 2024_2025.xlsx",
    "source": "F:\\project\\chatbot-ta\\chatbot-service\\data\\uploads\\fb0b2e1f-66cb-418a-8215-3204ca02d085_PUBLISH Mahasiswa - Perkuliahan Sem Genap 2024_2025.xlsx",
}

docs = [
    Document(
        page_content=f"{item['mata_kuliah']} diajarkan pada hari {item['hari']} pukul {item['jam']} di ruang {item['ruang']}.",
        metadata={**item, **extra_meta},
    )
    for item in jadwal_list
]

import asyncio


async def main():
    vector_store_service.initialize_embedding_model()
    vector_store_service.initialize_pg_vector("akademik")
    await vector_store_service.add_vectostore(docs, "akademik")


asyncio.run(main())

print("DONE")
