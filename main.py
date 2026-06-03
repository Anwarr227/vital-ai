from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import os


class Vitals(BaseModel):
    HR: float | None
    Temp: float | None
    O2Sat: float | None
    Resp: float | None

class PatientRecord(BaseModel):
    patient_id: int
    hour: int
    vitals: Vitals

class RiskResult(BaseModel):
    patient_id: int
    risk_level: str
    risk_score: int
    alerts: list[str]
    vitals: Vitals

class HealthStatus(BaseModel):
    status: str
    patients_loaded: int

class PatientList(BaseModel):
    patient_ids: list[int]
    total: int


BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "Prediction-of-sepsis", "test_data.csv")

# laad de csv en pakt per patient alleen de laatste meting
_df = pd.read_csv(CSV_PATH)
_latest = _df.sort_values("Hour").groupby("Patient_ID").last().reset_index()

_records: dict[int, pd.Series] = {
    int(r["Patient_ID"]): r for _, r in _latest.iterrows()
}

PATIENT_IDS: list[int] = sorted(_records.keys())


_THRESHOLDS = {
    "HR":    (60,   90),
    "Temp":  (36.0, 38.5),
    "O2Sat": (92,   float("inf")),  # only lower bound
    "Resp":  (12,   20),
}

_LABELS = {
    "HR":    "Hartslag",
    "Temp":  "Temperatuur",
    "O2Sat": "O2-saturatie",
    "Resp":  "Ademfrequentie",
}

_UNITS = {
    "HR":    "bpm",
    "Temp":  "°C",
    "O2Sat": "%",
    "Resp":  "/min",
}


# kijkt of een waarde buiten de normaalwaarden valt
def _is_abnormal(key: str, value: float) -> bool:
    lo, hi = _THRESHOLDS[key]
    if key == "O2Sat":
        return value < lo
    return value < lo or value > hi


# berekent het risico op basis van de vitale waarden en geeft een score terug
def assess_risk(row: pd.Series) -> RiskResult:
    alerts: list[str] = []
    score = 0

    for key in ("HR", "Temp", "O2Sat", "Resp"):
        val = row.get(key)
        if pd.isna(val):
            continue
        if _is_abnormal(key, float(val)):
            lo, hi = _THRESHOLDS[key]
            norm = f"≥{lo}" if key == "O2Sat" else f"{lo}–{hi}"
            alerts.append(
                f"{_LABELS[key]} afwijkend: {val:.1f} {_UNITS[key]} (normaal {norm} {_UNITS[key]})"
            )
            score += 1

    if score >= 2:
        level = "HIGH"
    elif score == 1:
        level = "MEDIUM"
    else:
        level = "LOW"

    def _f(k: str) -> float | None:
        v = row.get(k)
        return float(v) if pd.notna(v) else None

    return RiskResult(
        patient_id=int(row["Patient_ID"]),
        risk_level=level,
        risk_score=score,
        alerts=alerts,
        vitals=Vitals(HR=_f("HR"), Temp=_f("Temp"), O2Sat=_f("O2Sat"), Resp=_f("Resp")),
    )


app = FastAPI(title="Sepsis Risk API", version="1.0.0")

_static = os.path.join(BASE_DIR, "static")
if os.path.isdir(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/", include_in_schema=False)
async def root():
    p = os.path.join(_static, "index.html")
    return FileResponse(p) if os.path.isfile(p) else {"detail": "no frontend"}


# geeft aan of de api werkt en hoeveel patienten er zijn
@app.get("/health", response_model=HealthStatus)
async def health():
    return HealthStatus(status="ok", patients_loaded=len(PATIENT_IDS))


@app.get("/patients", response_model=PatientList)
async def list_patients():
    return PatientList(patient_ids=PATIENT_IDS, total=len(PATIENT_IDS))


@app.get("/patient/{patient_id}", response_model=PatientRecord)
async def get_patient(patient_id: int):
    if patient_id not in _records:
        raise HTTPException(404, detail=f"Patiënt {patient_id} niet gevonden")
    r = _records[patient_id]
    return PatientRecord(
        patient_id=patient_id,
        hour=int(r["Hour"]),
        vitals=Vitals(
            HR=float(r["HR"]) if pd.notna(r["HR"]) else None,
            Temp=float(r["Temp"]) if pd.notna(r["Temp"]) else None,
            O2Sat=float(r["O2Sat"]) if pd.notna(r["O2Sat"]) else None,
            Resp=float(r["Resp"]) if pd.notna(r["Resp"]) else None,
        ),
    )


# geeft het risico terug voor een patient
@app.get("/risk/{patient_id}", response_model=RiskResult)
async def get_risk(patient_id: int):
    if patient_id not in _records:
        raise HTTPException(404, detail=f"Patiënt {patient_id} niet gevonden")
    return assess_risk(_records[patient_id])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
