"""
Initialize the SQLite database for the ANPR system.

Creates `authorized_plates` per spec (Stage 7), seeds it with the MIPA UGM
roster, and creates `detection_logs` for in-DB transaction history (used by
the admin panel; the CSV at results/logs/verification_log.csv is the
per-run audit trail).

Run:
    python init_db.py            # create + seed if empty
    python init_db.py --reset    # drop and recreate (destroys data)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

from config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS authorized_plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT UNIQUE NOT NULL,
    vehicle_type TEXT NOT NULL CHECK(vehicle_type IN ('car','motorcycle')),
    owner_category TEXT NOT NULL CHECK(owner_category IN ('student','staff','faculty')),
    owner_name TEXT,
    registration_date TEXT NOT NULL,
    expiry_date TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_authorized_plate_number ON authorized_plates(plate_number);
CREATE INDEX IF NOT EXISTS idx_authorized_is_active ON authorized_plates(is_active);

CREATE TABLE IF NOT EXISTS detection_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    original_filename TEXT NOT NULL,
    detected_plate_raw TEXT,
    normalized_plate TEXT,
    ocr_engine_used TEXT,
    ocr_confidence REAL,
    detection_confidence REAL,
    verification_status TEXT NOT NULL,
    match_type TEXT,
    matched_plate TEXT,
    processing_time_ms REAL,
    error_category TEXT,
    annotated_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON detection_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_status ON detection_logs(verification_status);
"""


# (plate_number, vehicle_type, owner_category, owner_name, registration_date, expiry_date)
SEED: list[tuple[str, str, str, str, str, str]] = [
    ("AB 1234 CD", "motorcycle", "student", "Ahmad Fauzan", "2025-08-01", "2026-07-31"),
    ("AB 2345 EF", "motorcycle", "student", "Siti Nurhaliza", "2025-08-01", "2026-07-31"),
    ("AB 3456 GH", "motorcycle", "student", "Budi Santoso", "2025-08-01", "2026-07-31"),
    ("AB 4567 IJ", "motorcycle", "student", "Dewi Lestari", "2025-08-01", "2026-07-31"),
    ("AB 5678 KL", "motorcycle", "student", "Rizki Pratama", "2025-08-01", "2026-07-31"),
    ("AB 6789 MN", "motorcycle", "student", "Putri Ayu", "2025-08-01", "2026-07-31"),
    ("AB 7890 OP", "motorcycle", "student", "Andi Wijaya", "2025-08-01", "2026-07-31"),
    ("AB 1357 QR", "motorcycle", "student", "Rina Marlina", "2025-08-01", "2026-07-31"),
    ("AB 2468 ST", "motorcycle", "student", "Hendra Gunawan", "2025-08-01", "2026-07-31"),
    ("AB 3579 UV", "motorcycle", "student", "Nisa Fitriani", "2025-08-01", "2026-07-31"),
    ("AB 4680 WX", "motorcycle", "student", "Fajar Nugroho", "2025-08-01", "2026-07-31"),
    ("AB 1111 YZ", "motorcycle", "student", "Dian Permata", "2025-08-01", "2026-07-31"),
    ("AB 1001 AA", "car", "faculty", "Dr. Suryanto", "2025-01-01", "2027-12-31"),
    ("AB 1002 BB", "car", "faculty", "Prof. Hartono", "2025-01-01", "2027-12-31"),
    ("AB 1003 CC", "car", "faculty", "Dr. Wulandari", "2025-01-01", "2027-12-31"),
    ("AB 1004 DD", "car", "faculty", "Dr. Prasetyo", "2025-01-01", "2027-12-31"),
    ("AB 1005 EE", "car", "faculty", "Prof. Rahayu", "2025-01-01", "2027-12-31"),
    ("AB 1006 FF", "car", "faculty", "Dr. Setiawan", "2025-01-01", "2027-12-31"),
    ("AB 1194 XT", "car", "faculty", "Faculty Vehicle (test subject)", "2026-06-07", "2027-12-31"),
    ("AB 2001 GG", "motorcycle", "staff", "Bambang Hermawan", "2025-01-01", "2026-12-31"),
    ("AB 2002 HH", "motorcycle", "staff", "Sri Wahyuni", "2025-01-01", "2026-12-31"),
    ("AB 2003 II", "motorcycle", "staff", "Agus Riyadi", "2025-01-01", "2026-12-31"),
    ("AB 2004 JJ", "car", "staff", "Endang Susilowati", "2025-01-01", "2026-12-31"),
    ("AA 1234 AB", "motorcycle", "student", "Yoga Saputra", "2025-08-01", "2026-07-31"),
    ("AA 5678 CD", "motorcycle", "student", "Mega Putri", "2025-08-01", "2026-07-31"),
    ("AD 1234 EF", "motorcycle", "student", "Bayu Aji", "2025-08-01", "2026-07-31"),
    ("AD 5678 GH", "car", "faculty", "Retno Wati", "2025-01-01", "2027-12-31"),
    ("H 1234 AB", "motorcycle", "student", "Joko Susilo", "2025-08-01", "2026-07-31"),
    ("H 5678 CD", "car", "faculty", "Dr. Mulyono", "2025-01-01", "2027-12-31"),
    ("B 1234 EFG", "car", "faculty", "Ir. Sudirman", "2025-01-01", "2027-12-31"),
    ("D 1234 ABC", "car", "faculty", "Prof. Kurniawan", "2026-01-01", "2026-06-30"),
]


def init(reset: bool = False) -> None:
    if reset and DB_PATH.exists():
        print(f"[reset] removing {DB_PATH}")
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        existing = conn.execute("SELECT COUNT(*) FROM authorized_plates").fetchone()[0]
        if existing == 0:
            conn.executemany(
                "INSERT INTO authorized_plates "
                "(plate_number, vehicle_type, owner_category, owner_name, registration_date, expiry_date) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                SEED,
            )
            conn.commit()
            print(f"[seed] inserted {len(SEED)} authorized plates")
        else:
            print(f"[seed] skipped — {existing} authorized plates already present")
        print(f"[done] database ready at {DB_PATH}")
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize the ANPR SQLite database.")
    parser.add_argument("--reset", action="store_true", help="Drop existing DB before creating.")
    args = parser.parse_args()
    init(reset=args.reset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
