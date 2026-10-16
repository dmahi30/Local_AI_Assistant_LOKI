"""Registry managing explicitly supported deterministic commands for LOKI."""

from __future__ import annotations

from typing import Any

from .base import BaseCommand
from .builtin import (
    AppLauncherCommand,
    FileDeleteCommand,
    FileSearchCommand,
    SystemInfoCommand,
)


class CommandRegistry:
    """Central registry of deterministic commands."""

    def __init__(self, load_defaults: bool = True) -> None:
        self._commands: dict[str, BaseCommand] = {}
        if load_defaults:
            self.register(SystemInfoCommand())
            self.register(AppLauncherCommand())
            self.register(FileSearchCommand())
            self.register(FileDeleteCommand())

    def register(self, command: BaseCommand) -> None:
        """Register a command instance."""
        self._commands[command.name] = command

    def get(self, name: str) -> BaseCommand | None:
        """Lookup command by its identifier."""
        return self._commands.get(name)

    def list_commands(self) -> list[BaseCommand]:
        """Return all registered commands."""
        return list(self._commands.values())

    def find_command(self, text: str) -> tuple[BaseCommand, dict[str, Any]] | None:
        """Match input text against all registered command patterns."""
        for command in self._commands.values():
            if command.can_handle(text):
                args = command.extract_args(text)
                return command, args
        return None
