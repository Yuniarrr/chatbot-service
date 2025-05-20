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

dosen_mapping = {}
with open("./parse/dosen.txt", "r", encoding="utf-8") as f:
    for line in f:
        if ":" in line:
            inisial, nama = line.strip().split(":", 1)
            dosen_mapping[inisial.strip()] = nama.strip()


def gabung_nama_dosen(nama_dosen_list):
    if not nama_dosen_list:
        return "Tidak diketahui"
    elif len(nama_dosen_list) == 1:
        return nama_dosen_list[0]
    else:
        return ", ".join(nama_dosen_list[:-1]) + " dan " + nama_dosen_list[-1]


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

        # Cari semester dan dosen di tiap baris mk_info
        for info_line in mk_info:
            match = re.search(r"Semester\s+(.+?)\s*\|\s*(.+)", info_line)
            if match:
                semester = match.group(1).strip()
                dosen = match.group(2).strip()
                break

        # Kalau dosen kosong, coba ambil semester saja
        if dosen == "":
            for info_line in mk_info:
                match_alt = re.search(r"Semester\s+(\d+)", info_line)
                if match_alt:
                    semester = match_alt.group(1).strip()
                    break

        # Split dosen (bisa lebih dari satu)
        dosen_list = [d.strip() for d in dosen.split(",") if d.strip()]
        # Kalau inisial gak ada di mapping, tetap pakai inisial tsb
        nama_dosen_list = [dosen_mapping.get(d, d) for d in dosen_list]
        nama_dosen_str = gabung_nama_dosen(nama_dosen_list)

        jadwal_list.append(
            {
                "hari": current_day,
                "jam": jam,
                "ruang": ruang,
                "mata_kuliah": mata_kuliah,
                "semester": semester,
                "dosen": nama_dosen_str,
                "mk_info": mk_info,
            }
        )


with open("./parse/jadwal.json", "w", encoding="utf-8") as f:
    json.dump(jadwal_list, f, ensure_ascii=False, indent=4)

extra_meta = {
    "file_id": "99811cf5-3f1e-4506-adfa-f545d696077e",
    "file_name": "99811cf5-3f1e-4506-adfa-f545d696077e_jadwal-mata-kuliah.xlsx",
    "source": "F:\\project\\chatbot-ta\\chatbot-service\\data\\uploads\\99811cf5-3f1e-4506-adfa-f545d696077e_jadwal-mata-kuliah.xlsx",
    "name": "jadwal-mata-kuliah.xlsx",
    "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "size": 80806,
    "collection_name": "perkuliahan",
    "document_type": "mata kuliah",
    "topik": "jadwal mata kuliah di departemen teknologi informasi",
    "tahun_ajaran": "2024/2025",
}

docs = [
    Document(
        page_content=f"{item['mata_kuliah']} untuk semester {item['semester']} diajarkan pada hari {item['hari']} pukul {item['jam']} di ruang {item['ruang']} oleh {item['dosen']}.",
        metadata={**item, **extra_meta},
    )
    for item in jadwal_list
]

import asyncio


async def main():
    vector_store_service.initialize_embedding_model()
    vector_store_service.initialize_pg_vector("perkuliahan")
    await vector_store_service.add_vectostore(docs, "perkuliahan")


asyncio.run(main())

print("DONE")
