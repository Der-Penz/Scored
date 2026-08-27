from __future__ import annotations

import threading
import weakref
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Subscription:
    """Handle returned by EventChannel.subscribe(); call cancel() to unsubscribe."""

    __slots__ = ("_channel", "_event_type", "_callback_ref", "_cancelled")

    def __init__(
        self,
        channel: EventChannel,
        event_type: type,
        callback_ref: weakref.ref[Callable[..., Any]],
    ) -> None:
        self._channel = channel
        self._event_type = event_type
        self._callback_ref = callback_ref
        self._cancelled = False

    def cancel(self) -> None:
        """Unsubscribe this callback from the event channel."""
        if self._cancelled:
            return
        self._cancelled = True
        self._channel._unsubscribe(self._event_type, self._callback_ref)


class EventChannel:
    """A single typed event bus. Controllers emit and subscribe without referencing each other."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # event_type -> list of weakrefs to callbacks
        self._subscribers: dict[type, list[weakref.ref[Callable[..., Any]]]] = {}

    def subscribe(self, event_type: type[T], callback: Callable[[T], None]) -> Subscription:
        """
        Register *callback* to be called whenever an event of *event_type* is emitted.

        Returns a Subscription handle; call .cancel() to unsubscribe.
        """
        ref = weakref.ref(callback)
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(ref)
        return Subscription(self, event_type, ref)

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
            for subscribed_type, refs in self._subscribers.items():
                if issubclass(event_type, subscribed_type):
                    # Prune dead refs while we iterate
                    alive: list[weakref.ref[Callable[..., Any]]] = []
                    for ref in refs:
                        cb = ref()
                        if cb is not None:
                            alive.append(ref)
                            callbacks.append(cb)
                    self._subscribers[subscribed_type] = alive
        for cb in callbacks:
            cb(event)

    def _unsubscribe(self, event_type: type, callback_ref: weakref.ref) -> None:
        """Remove a specific callback reference from the subscriber list."""
        with self._lock:
            refs = self._subscribers.get(event_type)
            if refs is not None:
                try:
                    refs.remove(callback_ref)
                except ValueError:
                    pass
