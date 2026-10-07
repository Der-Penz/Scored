import tkinter as tk
from typing import TypeVar

import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.model.layout import Layout, Panel, load_layout
from gui.view.dartboard_view import DartboardView
from gui.view.dartgame_view import DartGameView
from gui.view.panel_manager import PanelManager
from gui.view.source_view import SourceView

T = TypeVar("T")


class AppView(ttk.Frame):
    """
    The main application view that contains all other views.

    The window is a menu bar followed by a :class:`PanelManager`, which decides
    where each of the dartboard, camera feed and control panel is shown: in the
    left or right column, in a window of its own, or not at all.
    """

    def __init__(self, master: tk.Tk):
        super().__init__(master, style="Card.TFrame", padding=0)
        self.master = master

        self._event_channel = EventChannel()

        self.create_widgets()

    @property
    def event_channel(self) -> EventChannel:
        """Bus shared with the controllers for panel lifecycle notifications."""
        return self._event_channel

    def create_widgets(self) -> None:
        """Build the menu bar and the panel area."""
        self.menu_frame = ttk.Frame(self.master)
        self.menu_frame.pack(fill="x", side="top")

        self.menu_separator = ttk.Separator(self.master, orient="horizontal")
        self.menu_separator.pack(fill="x", side="top")

        self.pack(fill="both", expand=True)

        self.panel_manager = PanelManager(
            self,
            self._event_channel,
            {
                Panel.DARTBOARD: DartboardView,
                Panel.SOURCE: SourceView,
                Panel.CONTROL: DartGameView,
            },
        )
        self.panel_manager.pack(fill="both", expand=True)
        self.panel_manager.apply(load_layout())

    @property
    def layout(self) -> Layout:
        """Return the layout currently on screen."""
        return self.panel_manager.layout

    def apply_layout(self, layout: Layout) -> None:
        """Dock the panels according to *layout*."""
        self.panel_manager.apply(layout)

    def _panel_view(self, panel: Panel, kind: type[T]) -> T:
        """Return the live widget of *panel*, which is rebuilt when it changes host."""
        widget = self.panel_manager.widget(panel)
        if not isinstance(widget, kind):
            raise TypeError(
                f"Panel {panel.label} is a {type(widget).__name__}, not a {kind.__name__}"
            )
        return widget

    @property
    def dartboard_view(self) -> DartboardView:
        """The live dartboard panel."""
        return self._panel_view(Panel.DARTBOARD, DartboardView)

    @property
    def source_view(self) -> SourceView:
        """The live camera feed panel."""
        return self._panel_view(Panel.SOURCE, SourceView)

    @property
    def game_view(self) -> DartGameView:
        """The live control panel."""
        return self._panel_view(Panel.CONTROL, DartGameView)
