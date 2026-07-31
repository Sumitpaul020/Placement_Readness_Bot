"""
tools.py
--------
The two agent tools required by the capstone brief:
  1. predict_placement -> wraps the trained DecisionTreeClassifier
  2. search_policy      -> wraps the RAG pipeline over the policy document
"""

import joblib
from langchain_core.tools import tool
from rag_utils import search_policy_raw

MODEL_PATH = "model/placement_model.pkl"

_model_bundle = None


def _load_model():
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


@tool
def predict_placement(
    cgpa: float,
    backlogs: int,
    internships: int,
    projects: int,
    coding_score: float,
    communication_score: float,
    attendance_percent: float,
) -> str:
    """Predict a student's placement likelihood using the trained ML model.

    Use this tool ONLY when the user gives (or you can reasonably infer)
    concrete numeric details about themselves: CGPA, number of active
    backlogs, number of internships, number of projects, coding_score
    (0-100), communication_score (0-100), and attendance_percent (0-100).

    Args:
        cgpa: Student's current CGPA, out of 10.
        backlogs: Number of currently active/standing backlogs.
        internships: Number of internships completed.
        projects: Number of academic/personal projects completed.
        coding_score: Self-rated or test-based coding skill score, 0-100.
        communication_score: Self-rated or test-based communication score, 0-100.
        attendance_percent: Current attendance percentage, 0-100.

    Returns:
        A short string with the predicted label (Placed / Not Placed) and
        the model's estimated probability of placement.
    """
    bundle = _load_model()
    model = bundle["model"]
    features = bundle["features"]

    row = {
        "cgpa": cgpa,
        "backlogs": backlogs,
        "internships": internships,
        "projects": projects,
        "coding_score": coding_score,
        "communication_score": communication_score,
        "attendance_percent": attendance_percent,
    }
    X = [[row[f] for f in features]]

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # probability of class "1" = placed

    label = "Likely to be PLACED" if pred == 1 else "Currently NOT likely to be placed"
    return (
        f"{label}. Model estimated placement probability: {proba*100:.1f}%. "
        f"(Based on: CGPA={cgpa}, backlogs={backlogs}, internships={internships}, "
        f"projects={projects}, coding_score={coding_score}, "
        f"communication_score={communication_score}, attendance={attendance_percent}%)"
    )


@tool
def search_policy(query: str) -> str:
    """Search the official placement eligibility/policy document to answer
    questions about rules, eligibility criteria, documentation, debarment,
    dress code, backlog policy, number-of-offers rules, etc.

    Use this tool for any question about official RULES or POLICY - not for
    predicting whether a specific student will get placed (use
    predict_placement for that).

    Args:
        query: The user's question about placement policy/eligibility.

    Returns:
        Relevant policy text if found, or an honest statement that the
        document does not cover this question.
    """
    result = search_policy_raw(query)
    if result == "NOT_FOUND":
        return (
            "I couldn't find anything about this in the placement policy "
            "document I have access to. This might be covered in a separate "
            "circular, or you should confirm with the Placement Cell directly."
        )
    return f"Relevant policy excerpt(s):\n{result}"
