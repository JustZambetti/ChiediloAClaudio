"""Pydantic data models for the AI personal trainer."""
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


class AthleteProfile(BaseModel):
    version: int = 1
    name: str
    body_weight_kg: float
    height_cm: float
    years_climbing: int
    boulder_grade: str
    sport_grade: str
    training_days_per_week: int = Field(ge=2, le=7)
    session_duration_minutes: int = Field(ge=30, le=180)
    equipment: list[str] = Field(default_factory=list)
    injuries: list[str] = Field(default_factory=list)
    primary_goal: str
    program_weeks: int = Field(default=8, ge=4, le=24)
    setup_date: str


class PerformanceTest(BaseModel):
    test_date: str
    fsi_kg: Optional[float] = None
    pullup_max_reps: Optional[int] = None
    pullup_weighted_kg: Optional[float] = None
    endurance_seconds: Optional[int] = None
    muscle_up_level: Optional[int] = Field(default=None, ge=0, le=4)
    pistol_level: Optional[int] = Field(default=None, ge=0, le=4)
    notes: str = ""


SessionType = Literal["strength", "finger", "climbing_tech", "endurance", "deload", "rest"]


class Exercise(BaseModel):
    name: str
    sets: int
    reps: str
    load: str
    rest_seconds: int
    notes: str = ""


class WorkoutDay(BaseModel):
    day_label: str
    session_type: SessionType
    focus: str
    duration_minutes: int
    exercises: list[Exercise]
    warmup: str = ""
    cooldown: str = ""


class TrainingWeek(BaseModel):
    week_number: int
    is_deload: bool = False
    theme: str
    days: list[WorkoutDay]
    test_scheduled: bool = False
    coaching_notes: str = ""


class TrainingProgram(BaseModel):
    version: int = 1
    sport: str = "climbing_calisthenics"
    total_weeks: int
    current_week: int = 1
    created_date: str
    last_adjusted_date: Optional[str] = None
    weeks: list[TrainingWeek]

    def current(self) -> TrainingWeek:
        idx = min(self.current_week - 1, len(self.weeks) - 1)
        return self.weeks[idx]


class SetLog(BaseModel):
    set_num: int
    reps_done: str
    load_used: str
    rir: Optional[int] = None
    notes: str = ""


class ExerciseLog(BaseModel):
    name: str
    sets: list[SetLog]
    skipped: bool = False
    skip_reason: str = ""


class WorkoutLog(BaseModel):
    log_date: str
    week_number: int
    day_label: str
    session_type: SessionType
    planned_day: WorkoutDay
    exercises: list[ExerciseLog]
    duration_minutes: int
    session_rpe: Optional[float] = Field(default=None, ge=1.0, le=10.0)
    notes: str = ""
    completed: bool = True


class ProgressSnapshot(BaseModel):
    weeks_completed: int
    total_sessions: int
    compliance_rate: float
    latest_test: Optional[PerformanceTest] = None
    previous_test: Optional[PerformanceTest] = None
    recent_logs: list[WorkoutLog]
    trend_notes: list[str]


class AdjustmentDecision(BaseModel):
    action: Literal[
        "continue",
        "increase_load",
        "deload_now",
        "reduce_volume",
        "add_endurance",
        "increase_skill_complexity",
        "full_replan",
    ]
    reason: str
    changes: list[str]
    updated_weeks: Optional[list[TrainingWeek]] = None
    coaching_message: str
