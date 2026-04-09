"""Feature hooks for capturing student transformer tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlockFeatureHook:
    """Capture the latest output of a transformer block."""

    latest: torch.Tensor | None = None
    handle: object | None = None

    def attach(self, module) -> None:
        """Attach hook to a module."""

        def _hook(_module, _inputs, output):
            self.latest = output

        self.handle = module.register_forward_hook(_hook)

    def clear(self) -> None:
        """Clear cached output."""
        self.latest = None

    def remove(self) -> None:
        """Detach the hook if attached."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
