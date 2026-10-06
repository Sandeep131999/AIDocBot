from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import AsyncIterator
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.auth import AuthUser, Role, get_current_user
from src.config import Config
from src.retrieval.vector_store import (
    get_vector_store,
    reset_project_scope,
    set_project_scope,
)

router = APIRouter(prefix="/api", tags=["Projects"])


@dataclass(frozen=True)
class ProjectContext:
    project_id: str
    user: AuthUser
    role: Role


class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class AddMemberRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)
    role: Role = Role.USER


def _store():
    try:
        return get_vector_store()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Project database is unavailable") from exc


async def get_project_context(
    request: Request,
    user: AuthUser = Depends(get_current_user),
) -> AsyncIterator[ProjectContext]:
    if not Config.AUTH_ENABLED:
        context = ProjectContext(project_id="default", user=user, role=Role.ADMIN)
    else:
        store = _store()
        projects = store.list_user_projects(user.user_id)
        requested_id = request.headers.get(Config.PROJECT_HEADER_NAME)
        if not requested_id:
            if len(projects) > 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Select a project using the {Config.PROJECT_HEADER_NAME} header",
                )
            if not projects:
                raise HTTPException(status_code=403, detail="No project access is configured")
            project = projects[0]
        else:
            project = next((item for item in projects if item["id"] == requested_id), None)
            if project is None:
                raise HTTPException(status_code=403, detail="You do not have access to this project")
        context = ProjectContext(
            project_id=project["id"],
            user=user,
            role=Role(project["role"]),
        )

    token = set_project_scope(context.project_id)
    try:
        yield context
    finally:
        reset_project_scope(token)


def require_project_role(*allowed: Role):
    async def dependency(
        context: ProjectContext = Depends(get_project_context),
    ) -> ProjectContext:
        if context.role not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient project role")
        return context
    return dependency


@router.get("/projects")
async def list_projects(user: AuthUser = Depends(get_current_user)):
    if not Config.AUTH_ENABLED:
        return {"projects": [{"id": "default", "name": "Default project", "role": "admin"}]}
    return {"projects": _store().list_user_projects(user.user_id)}


@router.post("/projects", status_code=201)
async def create_project(
    body: CreateProjectRequest,
    user: AuthUser = Depends(get_current_user),
):
    project_id = str(uuid.uuid4())
    _store().create_project(project_id, body.name.strip(), user.user_id)
    return {"id": project_id, "name": body.name.strip(), "role": "admin"}


@router.get("/projects/{project_id}/members")
async def list_project_members(
    project_id: str,
    user: AuthUser = Depends(get_current_user),
):
    store = _store()
    if not Config.AUTH_ENABLED:
        return {"members": []}
    if not store.get_project_role(project_id, user.user_id):
        raise HTTPException(status_code=403, detail="You do not have access to this project")
    return {"members": store.list_project_members(project_id)}


@router.post("/projects/{project_id}/members", status_code=201)
async def add_project_member(
    project_id: str,
    body: AddMemberRequest,
    user: AuthUser = Depends(get_current_user),
):
    store = _store()
    role = store.get_project_role(project_id, user.user_id) if Config.AUTH_ENABLED else "admin"
    if role != "admin":
        raise HTTPException(status_code=403, detail="Project admin role required")
    member_id = store.find_user_by_email_and_tenant(body.email.strip(), user.tenant_id)
    if not member_id:
        raise HTTPException(
            status_code=404,
            detail="User account must exist before being added",
        )
    store.add_project_member(project_id, member_id, body.role.value)
    return {"project_id": project_id, "user_id": member_id, "role": body.role.value}


@router.get("/monitoring/logs")
async def list_monitoring_logs(
    limit: int = 100,
    context: ProjectContext = Depends(get_project_context),
):
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    return {
        "logs": _store().list_question_logs(context.project_id, limit=limit),
        "project_id": context.project_id,
    }


@router.get("/monitoring/summary")
async def monitoring_summary(
    context: ProjectContext = Depends(get_project_context),
):
    store = _store()
    logs = store.list_question_logs(context.project_id, limit=500)
    return {
        "project_id": context.project_id,
        "requests": len(logs),
        "errors": sum(log["status"] != "success" for log in logs),
        "estimated_cost_usd": round(sum(log["estimated_cost_usd"] for log in logs), 6),
        "average_latency_ms": (
            round(sum(log["latency_ms"] for log in logs) / len(logs), 1) if logs else 0
        ),
    }
