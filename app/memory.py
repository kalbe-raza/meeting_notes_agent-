"""In-memory bounded conversation history. Lost on restart (single worker)."""

from collections import defaultdict
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class Turn:
    role: str   # 'user' | 'assistant'
    content: str


class Memory:
    """Bounded per-session message history."""

    def __init__(self):
        self._store: dict[str, list[Turn]] = defaultdict(list)
        self._goals: dict[str, str] = {}

    def get(self, session_id: str) -> list[dict]:
        """Return history as list of dicts for prompt assembly."""
        turns = self._store.get(session_id, [])
        return [{'role': t.role, 'content': t.content} for t in turns]

    def add(self, session_id: str, user_msg: str, assistant_msg: str):
        """Append a user/assistant turn, trimming to max_history_messages."""
        self._store[session_id].append(Turn(role='user', content=user_msg))
        self._store[session_id].append(Turn(role='assistant', content=assistant_msg))

        # Trim: keep the most recent N turns.
        # Use >= so the list never grows beyond the cap even momentarily.
        max_turns = settings.max_history_messages
        if len(self._store[session_id]) >= max_turns:
            self._store[session_id] = self._store[session_id][-max_turns:]

    def set_goal(self, session_id: str, goal: str):
        self._goals[session_id] = goal

    def get_goal(self, session_id: str) -> str:
        return self._goals.get(session_id, '')

    def clear(self, session_id: str):
        self._store.pop(session_id, None)
        self._goals.pop(session_id, None)
