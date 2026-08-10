"""
data/real_corilla_loader.py
---------------------------
Interface for loading PRIVATE Corilla CSV data.

WARNING: This module accesses proprietary data. NEVER commit actual CSVs.

Usage:
    from data.real_corilla_loader import load_real_corilla_data
    data = load_real_corilla_data("path/to/private/csvs/")

The public synthetic pipeline does NOT import this module.
Enable with:  python training/train.py --data-source real --real-data-dir /path/
"""
from __future__ import annotations
import logging
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


def load_real_corilla_data(data_dir: str, cfg: dict) -> dict[str, pd.DataFrame]:
    """
    Load private Corilla CSVs and normalise to the project's standard schema.

    Expected files (columns):
        students.csv      : student_id, registration_date
        institutions.csv  : institution_id, name, accreditation_status
        certificates.csv  : certificate_id, student_id, institution_id,
                            issue_date, revocation_date
        verifiers.csv     : verifier_id, institution_id, authorized  (optional)

    Returns
    -------
    dict with keys: students, institutions, verifiers, credentials
    """
    p = Path(data_dir)
    required = ["students.csv", "institutions.csv", "certificates.csv"]
    for fname in required:
        if not (p / fname).exists():
            raise FileNotFoundError(
                f"Required Corilla file not found: {p / fname}\n"
                "Real Corilla data is private. Run with --data-source synthetic "
                "for the public reproducible pipeline."
            )

    # ── Students ─────────────────────────────────────────
    students = pd.read_csv(p / "students.csv")
    students = students.rename(columns={"student_id": "student_id",
                                         "registration_date": "registration_time"})
    ref_date = pd.to_datetime(students["registration_time"]).min()
    students["registration_time"] = (
        (pd.to_datetime(students["registration_time"]) - ref_date)
        .dt.total_seconds().astype(int)
    )
    students = students[["student_id", "registration_time"]].reset_index(drop=True)

    # ── Institutions ──────────────────────────────────────
    institutions = pd.read_csv(p / "institutions.csv")
    institutions = institutions.rename(
        columns={"institution_id": "institution_id",
                 "accreditation_status": "accredited"}
    )
    institutions["accredited"] = institutions["accredited"].astype(int)
    if "institution_type" not in institutions.columns:
        institutions["institution_type"] = "unknown"
    institutions["is_fake"] = 0
    institutions = institutions[
        ["institution_id", "accredited", "institution_type", "is_fake"]
    ].reset_index(drop=True)

    # ── Verifiers ─────────────────────────────────────────
    ver_path = p / "verifiers.csv"
    if not ver_path.exists():
        warnings.warn(
            "verifiers.csv not found in real data directory. "
            "Synthesising placeholder verifiers (5 per institution).",
            UserWarning, stacklevel=2,
        )
        rng = np.random.default_rng(0)
        rows = []
        vid = 0
        for inst_id in institutions["institution_id"]:
            for _ in range(5):
                rows.append({"verifier_id": vid, "institution_id": inst_id,
                              "authorized": 1, "is_corrupt": 0})
                vid += 1
        verifiers = pd.DataFrame(rows)
    else:
        verifiers = pd.read_csv(ver_path)
        verifiers["is_corrupt"] = 0
        verifiers = verifiers[["verifier_id", "institution_id", "authorized", "is_corrupt"]]

    # ── Credentials ───────────────────────────────────────
    certs = pd.read_csv(p / "certificates.csv")
    certs = certs.rename(columns={
        "certificate_id": "credential_id",
        "student_id": "student_id",
        "institution_id": "institution_id",
        "issue_date": "issue_time",
        "revocation_date": "revocation_time",
    })
    certs["issue_time"] = (
        (pd.to_datetime(certs["issue_time"]) - ref_date)
        .dt.total_seconds().fillna(0).astype(int)
    )
    has_revoke = certs["revocation_time"].notna()
    certs["revocation_time"] = (
        (pd.to_datetime(certs["revocation_time"]) - ref_date)
        .dt.total_seconds().fillna(-1).astype(int)
    )
    certs["status"] = np.where(has_revoke, "revoked", "active")
    credentials = certs[
        ["credential_id", "student_id", "institution_id",
         "issue_time", "revocation_time", "status"]
    ].reset_index(drop=True)

    log.info(
        "Real Corilla data loaded: %d students, %d institutions, "
        "%d verifiers, %d credentials",
        len(students), len(institutions), len(verifiers), len(credentials),
    )
    return dict(students=students, institutions=institutions,
                verifiers=verifiers, credentials=credentials)
