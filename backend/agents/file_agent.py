from backend.models.agent import AgentSpec, CapabilitySchema
from backend.models.task import Task, FailureType
from backend.agents.base_agent import BaseAgent


class FileAgent(BaseAgent):

    def run(self, task: Task, goal: str, context: dict) -> dict:
        goal_lower = goal.lower()

        # ── Read ────────────────────────────────────────────────
        if any(w in goal_lower for w in ["read", "open", "show", "display", "get content"]):
            path = context.get("path") or self._extract_path(goal)
            if not path:
                return self.failure("No file path provided for read operation")
            result = self.use_skill("read_file", {"path": path}, task)
            if result.success:
                return self.success(result.data, f"Read file: {path}")
            return self.failure(result.error or "Read failed", FailureType.SKILL)

        # ── List directory ──────────────────────────────────────
        if any(w in goal_lower for w in ["list", "directory", "folder", "ls", "dir"]):
            path = context.get("path") or self._extract_path(goal) or "."
            result = self.use_skill("list_directory", {"path": path}, task)
            if result.success:
                return self.success(result.data, f"Listed directory: {path}")
            return self.failure(result.error or "List failed", FailureType.SKILL)

        # ── Search ──────────────────────────────────────────────
        if any(w in goal_lower for w in ["search", "find", "locate"]):
            directory = context.get("directory", ".")
            pattern = context.get("pattern") or self._extract_pattern(goal)
            result = self.use_skill("search_files", {"directory": directory, "pattern": pattern}, task)
            if result.success:
                return self.success(result.data, f"Found {result.data.get('count', 0)} files")
            return self.failure(result.error or "Search failed", FailureType.SKILL)

        # ── Write ───────────────────────────────────────────────
        if any(w in goal_lower for w in ["write", "save", "create"]):
            path = context.get("path") or self._extract_path(goal)
            content = context.get("content", "")
            if not path:
                return self.failure("No file path provided for write operation")
            result = self.use_skill("write_file", {"path": path, "content": content}, task)
            if result.error == "AWAITING_CONFIRMATION":
                return self.awaiting_confirmation(
                    "write_file",
                    {"path": path, "content": content},
                    f"Write {len(content)} characters to '{path}'?"
                )
            if result.success:
                return self.success(result.data, f"Written to {path}")
            return self.failure(result.error or "Write failed", FailureType.SKILL)

        # ── Delete ──────────────────────────────────────────────
        if any(w in goal_lower for w in ["delete", "remove", "unlink"]):
            path = context.get("path") or self._extract_path(goal)
            if not path:
                return self.failure("No file path provided for delete operation")
            result = self.use_skill("delete_file", {"path": path}, task)
            if result.error == "AWAITING_CONFIRMATION":
                return self.awaiting_confirmation(
                    "delete_file",
                    {"path": path},
                    f"Permanently delete '{path}'?"
                )
            if result.success:
                return self.success(result.data, f"Deleted {path}")
            return self.failure(result.error or "Delete failed", FailureType.SKILL)

        # ── Move ────────────────────────────────────────────────
        if any(w in goal_lower for w in ["move", "rename"]):
            source = context.get("source") or self._extract_path(goal)
            destination = context.get("destination", "")
            if not source or not destination:
                return self.failure("Source and destination paths required for move")
            result = self.use_skill("move_file", {"source": source, "destination": destination}, task)
            if result.error == "AWAITING_CONFIRMATION":
                return self.awaiting_confirmation(
                    "move_file",
                    {"source": source, "destination": destination},
                    f"Move '{source}' to '{destination}'?"
                )
            if result.success:
                return self.success(result.data, f"Moved {source} to {destination}")
            return self.failure(result.error or "Move failed", FailureType.SKILL)

        return self.failure(f"FileAgent could not interpret goal: {goal}")

    def _extract_path(self, text: str) -> str:
        import re
        patterns = [
            r"['\"]([^'\"]+\.\w+)['\"]",
            r"(?:file|path|from|to|at)\s+(\S+\.\w+)",
            r"(\S+\.(?:txt|py|json|csv|md|log|pdf|docx|xlsx))",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_pattern(self, text: str) -> str:
        import re
        match = re.search(r"\*\.\w+|\S+\.\w+", text)
        return match.group(0) if match else "*.*"


def create_file_agent() -> FileAgent:
    spec = AgentSpec(
        name="file_agent",
        description="Handles all local filesystem operations including reading, writing, searching, moving, copying, and deleting files and directories",
        capability_schema=CapabilitySchema(
            domain="file",
            task_types=["read_file", "write_file", "delete_file", "search_files", "list_directory", "move_file", "copy_file"],
            input_formats=["text", "path"],
            output_formats=["text", "file", "list"],
        ),
        system_prompt=(
            "You are the File Agent, a specialist in local filesystem operations. "
            "You read, write, search, move, copy, and delete files and directories. "
            "You always confirm before writing, moving, or deleting. "
            "You never access files outside the user's specified paths without explicit permission."
        ),
        skills=[
            "read_file", "write_file", "append_file",
            "delete_file", "list_directory", "search_files",
            "move_file", "copy_file"
        ],
    )
    return FileAgent(spec)