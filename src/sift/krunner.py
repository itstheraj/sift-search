from __future__ import annotations
import asyncio
import os

BUS_NAME = "org.sift.krunner"
OBJECT_PATH = "/sift"
KIND_ICON = {
    "image": "image-x-generic",
    "audio": "audio-x-generic",
    "video": "video-x-generic",
    "pdf": "application-pdf",
    "html": "text-html",
    "docx": "x-office-document",
    "text": "text-x-generic",
}


def _build_interface():
    from dbus_next import Variant
    from dbus_next.service import ServiceInterface, method
    from .engine import SearchEngine
    from .open_hit import open_path

    class KrunnerInterface(ServiceInterface):
        def __init__(self):
            super().__init__("org.kde.krunner1")
            self.engine = SearchEngine.fts_only()
            self._matches: dict[str, tuple[str, int | None]] = {}

        @method()
        def Match(self, query: "s") -> "a(sssida{sv})":
            q = query.strip()
            if len(q) < 2:
                return []
            self._matches = {}
            out = []
            for h in self.engine.search(q, limit=15):
                mid = h.path
                self._matches[mid] = (h.path, h.start_ms)
                text = os.path.basename(h.path)
                subtext = h.snippet or h.path
                icon = KIND_ICON.get(h.kind, "text-x-generic")
                relevance = max(0.1, min(1.0, h.score * 10))
                props = {"subtext": Variant("s", subtext)}
                out.append([mid, text, icon, 100, relevance, props])
            return out

        @method()
        def Actions(self) -> "a(sss)":
            return []

        @method()
        def Run(self, matchId: "s", actionId: "s"):
            target = self._matches.get(matchId)
            if target:
                open_path(target[0], target[1])

        @method()
        def Teardown(self):
            self._matches = {}

    return KrunnerInterface()


async def _serve() -> None:
    from dbus_next.aio import MessageBus

    bus = await MessageBus().connect()
    bus.export(OBJECT_PATH, _build_interface())
    await bus.request_name(BUS_NAME)
    print(f"sift krunner service ready at {BUS_NAME} {OBJECT_PATH}")
    await asyncio.get_event_loop().create_future()


def run_krunner_service() -> int:
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass
    return 0
