"""
Stage 10 — Flask Web Interface for the ANPR system.

Routes:
    GET  /              upload zone (single + batch)
    POST /process       single-file upload → run pipeline, render result
    POST /batch         folder/multi-upload → run pipeline, render table
    GET  /download_log  stream results/logs/verification_log.csv
    GET  /annotated/<f> serve annotated image
    GET  /admin         password-gated panel (plate CRUD + history)
    POST /admin/login   set session cookie
    POST /admin/plates  add/remove authorized plates
    GET  /admin/history transaction log with date+status filters
"""

from __future__ import annotations

import csv
import io
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, abort, flash, jsonify, redirect, render_template, request,
    send_file, send_from_directory, session, url_for,
)
from werkzeug.utils import secure_filename

from config import (
    ADMIN_PASSWORD,
    ANNOTATED_DIR,
    ALLOWED_IMAGE_EXTS,
    ALLOWED_VIDEO_EXTS,
    CSV_LOG_PATH,
    DB_PATH,
    DEFAULT_MODEL_VARIANT,
    FLASK_DEBUG,
    FLASK_PORT,
    MAX_CONTENT_LENGTH,
    MODEL_VARIANTS,
    SECRET_KEY,
    UPLOADS_DIR,
)
from pipeline import (
    STATUS_AUTHORIZED, STATUS_UNAUTHORIZED, STATUS_UNCERTAIN,
    STATUS_OCR_FAILED, STATUS_NO_PLATE, PipelineResult,
)
from pipeline.pipeline import process_image


app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# ─────────────────────────── Helpers ───────────────────────────

ALLOWED_EXTS = ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def _save_upload(file_storage) -> Path:
    fname = secure_filename(file_storage.filename or "upload")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = UPLOADS_DIR / f"{ts}_{fname}"
    out.parent.mkdir(parents=True, exist_ok=True)
    file_storage.save(out)
    return out


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _require_admin():
    if not session.get("is_admin"):
        abort(403)


def _serialize_result(r: PipelineResult) -> dict:
    return {
        "source": Path(r.source_path).name,
        "frame_index": r.frame_index,
        "is_video_frame": r.is_video_frame,
        "status": r.status,
        "error_category": r.error_category,
        "error_message": r.error_message,
        "processing_time_ms": round(r.processing_time_ms, 1),
        "annotated_url": (
            url_for("annotated", filename=Path(r.annotated_path).name)
            if r.annotated_path else None
        ),
        "plates": [
            {
                "raw_text": p.raw_text,
                "normalized_text": p.normalized_text,
                "chosen_engine": p.chosen_engine,
                "ocr_confidence": round(p.ocr_confidence, 3),
                "detection_confidence": round(p.bbox.confidence, 3),
                "status": p.verification.status if p.verification else r.status,
                "match_type": p.verification.match_type if p.verification else "NONE",
                "matched_plate": (p.verification.matched_plate or "") if p.verification else "",
                "flag": (p.verification.flag or "") if p.verification else "",
            }
            for p in r.plates
        ],
    }


# ─────────────────────────── Routes ────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        variants=list(MODEL_VARIANTS.keys()),
        default_variant=DEFAULT_MODEL_VARIANT,
    )


@app.route("/process", methods=["POST"])
def process_single():
    f = request.files.get("file")
    if not f or not f.filename:
        flash("No file selected.", "error")
        return redirect(url_for("index"))
    if not _is_allowed(f.filename):
        flash(f"Unsupported file type: {f.filename}", "error")
        return redirect(url_for("index"))

    variant = request.form.get("variant", DEFAULT_MODEL_VARIANT)
    if variant not in MODEL_VARIANTS:
        variant = DEFAULT_MODEL_VARIANT

    saved = _save_upload(f)
    results = process_image(saved, variant=variant)
    return render_template(
        "result.html",
        results=[_serialize_result(r) for r in results],
        variant=variant,
    )


@app.route("/batch", methods=["POST"])
def process_batch_route():
    files = request.files.getlist("files")
    if not files:
        flash("No files selected.", "error")
        return redirect(url_for("index"))

    variant = request.form.get("variant", DEFAULT_MODEL_VARIANT)
    if variant not in MODEL_VARIANTS:
        variant = DEFAULT_MODEL_VARIANT

    all_results: list[PipelineResult] = []
    for f in files:
        if not f.filename or not _is_allowed(f.filename):
            continue
        saved = _save_upload(f)
        all_results.extend(process_image(saved, variant=variant))

    return render_template(
        "result.html",
        results=[_serialize_result(r) for r in all_results],
        variant=variant,
        batch=True,
    )


@app.route("/download_log")
def download_log():
    if not CSV_LOG_PATH.exists():
        flash("No log file yet.", "error")
        return redirect(url_for("index"))
    return send_file(
        CSV_LOG_PATH,
        as_attachment=True,
        download_name="verification_log.csv",
        mimetype="text/csv",
    )


@app.route("/annotated/<path:filename>")
def annotated(filename: str):
    return send_from_directory(ANNOTATED_DIR, filename)


# ─────────────────────────── Admin ─────────────────────────────

@app.route("/admin")
def admin():
    if not session.get("is_admin"):
        return render_template("admin_login.html")
    with _db() as conn:
        plates = conn.execute(
            "SELECT * FROM authorized_plates ORDER BY plate_number"
        ).fetchall()
    return render_template("admin.html", plates=plates)


@app.route("/admin/login", methods=["POST"])
def admin_login():
    pw = request.form.get("password", "")
    if pw == ADMIN_PASSWORD:
        session["is_admin"] = True
        return redirect(url_for("admin"))
    flash("Wrong password.", "error")
    return redirect(url_for("admin"))


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("index"))


@app.route("/admin/plates", methods=["POST"])
def admin_plates():
    _require_admin()
    action = request.form.get("action")
    if action == "add":
        plate = (request.form.get("plate_number") or "").strip().upper()
        vtype = request.form.get("vehicle_type") or "car"
        cat = request.form.get("owner_category") or "staff"
        owner = (request.form.get("owner_name") or "").strip()
        reg = request.form.get("registration_date") or datetime.now().strftime("%Y-%m-%d")
        exp = request.form.get("expiry_date") or ""
        try:
            with _db() as conn:
                conn.execute(
                    "INSERT INTO authorized_plates "
                    "(plate_number, vehicle_type, owner_category, owner_name, registration_date, expiry_date) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (plate, vtype, cat, owner, reg, exp or None),
                )
                conn.commit()
            flash(f"Added {plate}.", "ok")
        except sqlite3.IntegrityError as e:
            flash(f"Could not add plate: {e}", "error")
    elif action == "delete":
        pid = request.form.get("id")
        if pid:
            with _db() as conn:
                conn.execute("DELETE FROM authorized_plates WHERE id = ?", (pid,))
                conn.commit()
            flash("Removed plate.", "ok")
    return redirect(url_for("admin"))


@app.route("/admin/history")
def admin_history():
    _require_admin()
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    status = request.args.get("status", "")

    sql = "SELECT * FROM detection_logs WHERE 1=1"
    params: list = []
    if start:
        sql += " AND date(timestamp) >= date(?)"
        params.append(start)
    if end:
        sql += " AND date(timestamp) <= date(?)"
        params.append(end)
    if status:
        sql += " AND verification_status = ?"
        params.append(status)
    sql += " ORDER BY timestamp DESC LIMIT 500"

    with _db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return render_template(
        "history.html",
        rows=rows, start=start, end=end, status=status,
        statuses=[STATUS_AUTHORIZED, STATUS_UNAUTHORIZED, STATUS_UNCERTAIN, STATUS_OCR_FAILED, STATUS_NO_PLATE],
    )


@app.route("/admin/history.csv")
def admin_history_csv():
    _require_admin()
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM detection_logs ORDER BY timestamp DESC"
        ).fetchall()
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="detection_logs.csv",
    )


# JSON ping for sanity check. Useful for verifying which config the running
# Flask process actually has in memory (config.yaml changes don't auto-reload).
@app.route("/healthz")
def healthz():
    from config import DET_CONF, INPUT_SIZE, OCR_CONF_THRESH, FUZZY_MAX_DIST
    return jsonify(
        ok=True,
        db=str(DB_PATH),
        default_variant=DEFAULT_MODEL_VARIANT,
        input_size=list(INPUT_SIZE),
        detection_conf_threshold=DET_CONF,
        ocr_conf_threshold=OCR_CONF_THRESH,
        fuzzy_max_distance=FUZZY_MAX_DIST,
    )


if __name__ == "__main__":
    import os
    from pathlib import Path
    port = int(os.getenv("PORT", str(FLASK_PORT)))
    # Tell Werkzeug to also watch config.yaml — by default its reloader only
    # watches .py modules, so YAML edits silently stay stale in memory.
    extra_files = [str(Path(__file__).resolve().parent / "config.yaml")]
    app.run(debug=FLASK_DEBUG, port=port, host="0.0.0.0", extra_files=extra_files)
