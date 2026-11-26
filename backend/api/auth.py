"""
auth.py — Routes d'authentification JWT
========================================

Endpoints :
    POST /api/v1/auth/token    → Connexion (OAuth2 password flow)
    POST /api/v1/auth/register → Création de compte
    GET  /api/v1/auth/me       → Profil utilisateur connecté
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import crud
from backend.db.database import get_db
from backend.db.models import User
from backend.db.schemas import UserCreate, UserResponse
from backend.security.auth import create_access_token, get_current_user, verify_password

router = APIRouter()


@router.post("/token", summary="Obtenir un token JWT")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Authentifie l'utilisateur et retourne un bearer token JWT."""
    user = await crud.get_user_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Compte désactivé")

    token = create_access_token(data={"sub": user.username, "role": user.role})
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "role": user.role,
    }


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Enregistre un nouvel utilisateur (rôle par défaut : technician)."""
    try:
        user = await crud.create_user(db, payload)
        return user
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get("/me", response_model=UserResponse, summary="Profil utilisateur courant")
async def read_current_user(current_user: User = Depends(get_current_user)):
    """Retourne les informations de l'utilisateur authentifié."""
    return current_user
