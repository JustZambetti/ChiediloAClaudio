"""JSON-based persistence layer with atomic writes."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from trainer.schemas import AthleteProfile, PerformanceTest, TrainingProgram, WorkoutLog

DATA_DIR = Path("data")
LOGS_DIR = DATA_DIR / "logs"
TESTS_DIR = DATA_DIR / "tests"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    TESTS_DIR.mkdir(exist_ok=True)


def _atomic_write(path: Path, content: str) -> None:
    _ensure_dirs()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def is_setup_complete() -> bool:
    return (DATA_DIR / "profile.json").exists() and (DATA_DIR / "program.json").exists()


def load_profile() -> AthleteProfile:
    return AthleteProfile.model_validate_json((DATA_DIR / "profile.json").read_text())


def save_profile(p: AthleteProfile) -> None:
    _atomic_write(DATA_DIR / "profile.json", p.model_dump_json(indent=2))


def load_program() -> TrainingProgram:
    return TrainingProgram.model_validate_json((DATA_DIR / "program.json").read_text())


def save_program(prog: TrainingProgram) -> None:
    _atomic_write(DATA_DIR / "program.json", prog.model_dump_json(indent=2))


def save_log(log: WorkoutLog) -> None:
    _atomic_write(LOGS_DIR / f"{log.log_date}.json", log.model_dump_json(indent=2))


def load_log(date_str: str) -> Optional[WorkoutLog]:
    path = LOGS_DIR / f"{date_str}.json"
    if not path.exists():
        return None
    return WorkoutLog.model_validate_json(path.read_text())


def load_all_logs() -> list[WorkoutLog]:
    if not LOGS_DIR.exists():
        return []
    logs = []
    for f in sorted(LOGS_DIR.glob("*.json")):
        try:
            logs.append(WorkoutLog.model_validate_json(f.read_text()))
        except Exception:
            pass
    return logs


def load_recent_logs(n: int = 14) -> list[WorkoutLog]:
    return load_all_logs()[-n:]


def save_test(test: PerformanceTest) -> None:
    _atomic_write(TESTS_DIR / f"{test.test_date}.json", test.model_dump_json(indent=2))


def load_all_tests() -> list[PerformanceTest]:
    if not TESTS_DIR.exists():
        return []
    tests = []
    for f in sorted(TESTS_DIR.glob("*.json")):
        try:
            tests.append(PerformanceTest.model_validate_json(f.read_text()))
        except Exception:
            pass
    return tests
