import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config import settings
from backend.brain.orchestrator import brain
from backend.brain.session import SessionContext
from backend.factory.skill_factory import get_skill_factory
from backend.factory.agent_factory import get_agent_factory
from backend.registry.agent_registry import agent_registry
from backend.registry.skill_registry import skill_registry
from backend.skills.executor import skill_executor
import structlog

log = structlog.get_logger()

# ── Session store (in-memory for Phase 1) ────────────────────────
sessions: dict[str, SessionContext] = {}


# ── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AEGIS starting up", provider=settings.llm_provider, model=settings.ollama_model)
    log.info("Agents loaded", count=agent_registry.count())
    log.info("Skills loaded", count=skill_registry.count())
    yield
    log.info("AEGIS shutting down")


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="AEGIS",
    description="Adaptive Executive General Intelligence System",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────
class TaskRequest(BaseModel):
    input: str
    session_id: Optional[str] = None


class ConfirmRequest(BaseModel):
    session_id: str
    task_id: str
    skill_name: str
    params: dict
    confirmed: bool


class SkillPatchRequest(BaseModel):
    agent_name: str
    skills: list[str]


# ── Routes ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "AEGIS",
        "version": "3.0.0",
        "status": "online",
        "agents": agent_registry.count(),
        "skills": skill_registry.count(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
  
  
  
# ── Factory endpoints ─────────────────────────────────────────────
@app.post("/factory/skill")
def create_skill(body: dict):
    capability = body.get("capability", "")
    if not capability:
        raise HTTPException(status_code=400, detail="capability is required")
    factory = get_skill_factory(brain._llm_call)
    success, message, spec = factory.create(capability)
    return {
        "success": success,
        "message": message,
        "skill": spec.model_dump() if spec else None,
    }


@app.post("/factory/agent")
def create_agent(body: dict):
    capability = body.get("capability", "")
    if not capability:
        raise HTTPException(status_code=400, detail="capability is required")
    factory = get_agent_factory(brain._llm_call)
    success, message, spec = factory.create(capability)
    return {
        "success": success,
        "message": message,
        "agent": spec.model_dump() if spec else None,
    }


# ── Task endpoint — main entry point ─────────────────────────────
@app.post("/task")
def process_task(request: TaskRequest):
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = SessionContext(session_id=session_id)
    session = sessions[session_id]

    log.info("Task received", session_id=session_id, input=request.input[:80])

    result = brain.process(request.input, session)

    # Persist updated session
    sessions[session_id] = result.get("session", session)

    # Strip session from response (not serialisable as-is)
    response = {k: v for k, v in result.items() if k != "session"}
    response["session_id"] = session_id

    # Fix status for awaiting confirmation
    if result.get("awaiting_confirmation"):
      response["status"] = "awaiting_confirmation"

    return response


# ── Confirmation endpoint ────────────────────────────────────────
@app.post("/confirm")
def confirm_action(request: ConfirmRequest):
    if not request.confirmed:
        log.info("Action cancelled by user", skill=request.skill_name)
        return {"status": "cancelled", "message": "Action cancelled."}

    result = skill_executor.confirm_and_execute(
        request.skill_name,
        request.params,
        task_id=request.task_id,
    )

    return {
        "status": "completed" if result.success else "failed",
        "skill": request.skill_name,
        "success": result.success,
        "data": result.data,
        "error": result.error,
    }


# ── Agent endpoints ──────────────────────────────────────────────
@app.get("/agents")
def list_agents():
    agents = agent_registry.get_active()
    return {
        "count": len(agents),
        "agents": [
            {
                "name": a.name,
                "description": a.description,
                "skills": a.skills,
                "performance_score": a.performance_score,
                "task_count": a.task_count,
                "version": a.version,
            }
            for a in agents
        ],
    }


@app.get("/agents/{name}")
def get_agent(name: str):
    agent = agent_registry.get(name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    return agent.model_dump()


@app.post("/agents/{name}/patch-skills")
def patch_agent_skills(name: str, request: SkillPatchRequest):
    # Validate all skills exist
    missing = [s for s in request.skills if not skill_registry.exists(s)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Skills not found: {missing}")

    success = agent_registry.patch_skills(name, request.skills)
    if not success:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

    agent = agent_registry.get(name)
    return {
        "status": "patched",
        "agent": name,
        "skills": agent.skills,
        "version": agent.version,
    }


# ── Skill endpoints ───────────────────────────────────────────────
@app.get("/skills")
def list_skills():
    skills = skill_registry.get_all()
    return {
        "count": len(skills),
        "skills": [
            {
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "operation_type": s.operation_type,
                "requires_confirmation": s.requires_confirmation,
                "version": s.version,
            }
            for s in skills
        ],
    }


@app.get("/skills/{name}")
def get_skill(name: str):
    skill = skill_registry.get(name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    return skill.model_dump()


# ── Session endpoints ─────────────────────────────────────────────
@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "turn_count": session.turn_count,
        "last_agent": session.last_agent,
        "history": session.get_recent_context(10),
    }


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


# ── Registry endpoints ────────────────────────────────────────────
@app.get("/registry/status")
def registry_status():
    return {
        "agents": agent_registry.count(),
        "skills": skill_registry.count(),
        "active_sessions": len(sessions),
    }