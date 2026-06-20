"""
AEGIS memory layer — public API.

The Brain imports from here. Internal modules stay hidden.

    from backend.memory import (
        TaskRecord, TaskOutcome, get_task_store,
        get_preference_engine, get_agent_scorer, get_evolution_engine,
    )
"""

from .schemas import QualitySignals, TaskOutcome, TaskRecord
from .task_store import TaskStore, get_task_store
from .preference_engine import PreferenceEngine, UserProfile, get_preference_engine
from .agent_scorer import AgentScorer, ScoreBreakdown, get_agent_scorer
from .evolution_engine import EvolutionEngine, EvolutionOutcome, EvolutionResult, get_evolution_engine
from .knowledge_store import KnowledgeStore, SearchResult, get_knowledge_store
from .multi_agent import (
    WorkflowOrchestrator, WorkflowPlan, WorkflowStep,
    WorkflowResult, StepStatus, build_plan_from_routing,
)

__all__ = [
    # schemas
    "TaskRecord",
    "TaskOutcome",
    "QualitySignals",
    # task store
    "TaskStore",
    "get_task_store",
    # preference engine
    "PreferenceEngine",
    "UserProfile",
    "get_preference_engine",
    # agent scorer
    "AgentScorer",
    "ScoreBreakdown",
    "get_agent_scorer",
    # evolution engine
    "EvolutionEngine",
    "EvolutionOutcome",
    "EvolutionResult",
    "get_evolution_engine",
    # knowledge store
    "KnowledgeStore",
    "SearchResult",
    "get_knowledge_store",
    # multi-agent workflows
    "WorkflowOrchestrator",
    "WorkflowPlan",
    "WorkflowStep",
    "WorkflowResult",
    "StepStatus",
    "build_plan_from_routing",
]
