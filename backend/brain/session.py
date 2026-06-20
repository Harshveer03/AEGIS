from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SessionContext:
    session_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    turn_count: int = 0
    last_task_id: Optional[str] = None
    last_agent: Optional[str] = None
    last_result: Optional[dict] = None
    preferences: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def add_turn(self, task_id: str, agent: str, goal: str, result: dict):
        self.turn_count += 1
        self.last_task_id = task_id
        self.last_agent = agent
        self.last_result = result
        self.history.append({
            "turn": self.turn_count,
            "task_id": task_id,
            "agent": agent,
            "goal": goal,
            "success": result.get("success", False),
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep last 20 turns only
        if len(self.history) > 20:
            self.history = self.history[-20:]

    def get_recent_context(self, n: int = 3) -> list[dict]:
        return self.history[-n:] if self.history else []