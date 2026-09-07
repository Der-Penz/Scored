from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Subscription:
    """Handle returned by EventChannel.subscribe(); call cancel() to unsubscribe."""

    __slots__ = ("_channel", "_event_type", "_callback", "_cancelled")

    def __init__(
        self,
        channel: EventChannel,
        event_type: type,
        callback: Callable[..., Any],
    ) -> None:
        self._channel = channel
        self._event_type = event_type
        self._callback = callback
        self._cancelled = False

    def cancel(self) -> None:
        """Unsubscribe this callback from the event channel."""
        if self._cancelled:
            return
        self._cancelled = True
        self._channel._unsubscribe(self._event_type, self._callback)


class EventChannel:
    """A single typed event bus. Controllers emit and subscribe without referencing each other."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[type, list[Callable[..., Any]]] = {}

    def subscribe(
        self, event_type: type[T], callback: Callable[[T], None]
    ) -> Subscription:
        """
        Register *callback* to be called whenever an event of *event_type* is emitted.

        Returns a Subscription handle; call .cancel() to unsubscribe.
        """
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)
        return Subscription(self, event_type, callback)

    def emit(self, event: Any) -> None:
        """
        Emit *event* to all subscribers whose registered type matches
        ``type(event)`` via subclass check (so a subscriber for a base
        event type also receives subclass events).
        """
        event_type = type(event)
        # Collect matching callbacks under the lock, invoke outside it
        callbacks: list[Callable[..., Any]] = []
        with self._lock:
            for subscribed_type, cbs in self._subscribers.items():
                if issubclass(event_type, subscribed_type):
                    callbacks.extend(cbs)
        for cb in callbacks:
            cb(event)

    def _unsubscribe(self, event_type: type, callback: Callable[..., Any]) -> None:
        """Remove a specific callback from the subscriber list."""
        with self._lock:
            cbs = self._subscribers.get(event_type)
            if cbs is not None:
                try:
                    cbs.remove(callback)
                except ValueError:
                    pass
