-- ANPR System Database Setup
-- MIPA UGM Parking Lot

CREATE DATABASE IF NOT EXISTS anpr_mipa_ugm
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE anpr_mipa_ugm;

-- ──────────────────────────────────────────────
-- Authorized Vehicles Table
-- ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS authorized_vehicles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(15) NOT NULL UNIQUE,
    vehicle_type ENUM('motorcycle', 'car', 'other') NOT NULL DEFAULT 'motorcycle',
    owner_name VARCHAR(100) NOT NULL,
    owner_category ENUM('student', 'lecturer', 'staff', 'guest') NOT NULL,
    faculty VARCHAR(100) DEFAULT 'MIPA',
    registration_date DATE NOT NULL,
    expiry_date DATE NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_plate_number (plate_number),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB;

-- ──────────────────────────────────────────────
-- Detection Logs Table
-- ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS detection_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_filename VARCHAR(255) NOT NULL,
    detected_plate VARCHAR(15),
    easyocr_text VARCHAR(15),
    easyocr_confidence FLOAT,
    tesseract_text VARCHAR(15),
    tesseract_confidence FLOAT,
    final_text VARCHAR(15),
    final_confidence FLOAT,
    verification_status ENUM('AUTHORIZED', 'UNAUTHORIZED', 'UNCERTAIN', 'OCR_FAILED') NOT NULL,
    matched_plate VARCHAR(15),
    processing_time_ms INT,
    model_variant VARCHAR(20),
    preprocessing_enabled BOOLEAN DEFAULT TRUE,
    INDEX idx_timestamp (timestamp),
    INDEX idx_verification_status (verification_status)
) ENGINE=InnoDB;

-- ──────────────────────────────────────────────
-- Seed Data: ~30 Yogyakarta-area plates
-- ──────────────────────────────────────────────
INSERT INTO authorized_vehicles (plate_number, vehicle_type, owner_name, owner_category, faculty, registration_date, expiry_date) VALUES
-- Students (motorcycles)
('AB 1234 CD', 'motorcycle', 'Ahmad Fauzan', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 2345 EF', 'motorcycle', 'Siti Nurhaliza', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 3456 GH', 'motorcycle', 'Budi Santoso', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 4567 IJ', 'motorcycle', 'Dewi Lestari', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 5678 KL', 'motorcycle', 'Rizki Pratama', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 6789 MN', 'motorcycle', 'Putri Ayu', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 7890 OP', 'motorcycle', 'Andi Wijaya', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 1357 QR', 'motorcycle', 'Rina Marlina', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 2468 ST', 'motorcycle', 'Hendra Gunawan', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 3579 UV', 'motorcycle', 'Nisa Fitriani', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 4680 WX', 'motorcycle', 'Fajar Nugroho', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AB 1111 YZ', 'motorcycle', 'Dian Permata', 'student', 'MIPA', '2025-08-01', '2026-07-31'),

-- Lecturers (cars)
('AB 1001 AA', 'car', 'Dr. Suryanto', 'lecturer', 'MIPA', '2025-01-01', '2027-12-31'),
('AB 1002 BB', 'car', 'Prof. Hartono', 'lecturer', 'MIPA', '2025-01-01', '2027-12-31'),
('AB 1003 CC', 'car', 'Dr. Wulandari', 'lecturer', 'MIPA', '2025-01-01', '2027-12-31'),
('AB 1004 DD', 'car', 'Dr. Prasetyo', 'lecturer', 'MIPA', '2025-01-01', '2027-12-31'),
('AB 1005 EE', 'car', 'Prof. Rahayu', 'lecturer', 'MIPA', '2025-01-01', '2027-12-31'),
('AB 1006 FF', 'car', 'Dr. Setiawan', 'lecturer', 'MIPA', '2025-01-01', '2027-12-31'),

-- Staff (motorcycles)
('AB 2001 GG', 'motorcycle', 'Bambang Hermawan', 'staff', 'MIPA', '2025-01-01', '2026-12-31'),
('AB 2002 HH', 'motorcycle', 'Sri Wahyuni', 'staff', 'MIPA', '2025-01-01', '2026-12-31'),
('AB 2003 II', 'motorcycle', 'Agus Riyadi', 'staff', 'MIPA', '2025-01-01', '2026-12-31'),
('AB 2004 JJ', 'car', 'Endang Susilowati', 'staff', 'MIPA', '2025-01-01', '2026-12-31'),

-- Nearby region plates (common at UGM)
('AA 1234 AB', 'motorcycle', 'Yoga Saputra', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AA 5678 CD', 'motorcycle', 'Mega Putri', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AD 1234 EF', 'motorcycle', 'Bayu Aji', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('AD 5678 GH', 'car', 'Retno Wati', 'lecturer', 'MIPA', '2025-01-01', '2027-12-31'),
('H 1234 AB', 'motorcycle', 'Joko Susilo', 'student', 'MIPA', '2025-08-01', '2026-07-31'),
('H 5678 CD', 'car', 'Dr. Mulyono', 'lecturer', 'MIPA', '2025-01-01', '2027-12-31'),
('B 1234 EFG', 'car', 'Ir. Sudirman', 'lecturer', 'MIPA', '2025-01-01', '2027-12-31'),
('D 1234 ABC', 'car', 'Prof. Kurniawan', 'guest', 'MIPA', '2026-01-01', '2026-06-30');
