import os
import json
import logging
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(tags=["experiments"])

RESULTS_FILE_PATH = os.path.join("data", "experiments", "phase5_scifact_experiment_results.json")

@router.get("/experiments/scifact", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_scifact_experiment_results() -> Dict[str, Any]:
    """
    GET /api/experiments/scifact
    
    Returns the Phase 5 SciFact Controlled Experiment results stored in JSON format.
    Exposes measured metrics (Accuracy, Precision, Recall, F1, ROC-AUC) for both
    the Cosine Similarity Baseline and the Supervised Logistic Regression model.
    """
    file_path = RESULTS_FILE_PATH
    if not os.path.exists(file_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        alt_path = os.path.join(project_root, file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Phase 5 SciFact experiment results JSON file not found."
            )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error reading SciFact experiment results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load experiment results: {str(e)}"
        )
