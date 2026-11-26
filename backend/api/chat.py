"""
chat.py — Route de conversation avec l'agent LangGraph MaintiDoc
=================================================================

Endpoints :
    POST /api/v1/chat/ask        → Poser une question (JSON, image base64 optionnelle)
    POST /api/v1/chat/ask-upload → Poser une question avec upload multipart d'une image
"""

from __future__ import annotations

import base64
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from backend.agent.graph import run_agent
from backend.db.schemas import ChatRequest, ChatResponse
from backend.security.input_validator import (
    sanitize_text_output,
    validate_equipment_id,
    validate_question,
)

router = APIRouter()


@router.post("/ask", response_model=ChatResponse, summary="Interroger l'assistant MaintiDoc")
async def ask_agent(request: ChatRequest):
    """
    Exécute le workflow LangGraph pour une question textuelle (et image base64 optionnelle).
    """
    cleaned_question = validate_question(request.question)
    eq_id = validate_equipment_id(request.equipment_id) if request.equipment_id else None

    image_bytes: bytes | None = None
    if request.image_base64:
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Image base64 invalide")

    result = await run_agent(
        question=cleaned_question,
        document_ids=request.document_ids,
        equipment_id=eq_id,
        image_bytes=image_bytes,
        image_base64=request.image_base64,
    )

    safe_answer = sanitize_text_output(result.get("answer", ""))
    return ChatResponse(
        answer=safe_answer,
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
        requires_human_approval=result.get("requires_human_approval", False),
        ticket_draft=result.get("ticket_draft"),
    )


@router.post("/ask-upload", response_model=ChatResponse, summary="Question avec photo d'équipement")
async def ask_agent_with_image(
    question: str = Form(...),
    equipment_id: Optional[str] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    """
    Variante multipart/form-data permettant d'envoyer une photo directement depuis Streamlit.
    """
    cleaned_question = validate_question(question)
    eq_id = validate_equipment_id(equipment_id) if equipment_id else None

    image_bytes: bytes | None = None
    if image is not None:
        image_bytes = await image.read()
        logger.info(f"Image reçue pour analyse : {image.filename} ({len(image_bytes)} octets)")

    result = await run_agent(
        question=cleaned_question,
        equipment_id=eq_id,
        image_bytes=image_bytes,
    )

    return ChatResponse(
        answer=sanitize_text_output(result.get("answer", "")),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
        requires_human_approval=result.get("requires_human_approval", False),
        ticket_draft=result.get("ticket_draft"),
    )
