"""RCON client and test-server launcher for the Factorio test harness.

Lua runs only in the separate test save, to set up probes and read signals.
It never does model math.
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACTORIO = os.environ.get("FACTORIO", "/Applications/factorio.app/Contents/MacOS/factorio")   # the game binary (macOS default)
RCON_PORT = int(os.environ.get("FGPT_RCON_PORT", 27115))
GAME_PORT = int(os.environ.get("FGPT_GAME_PORT", 34297))
PASSWORD = os.environ.get("FGPT_RCON_PASSWORD", "fgpt-local")


class Rcon:
    def __init__(self, host: str = "127.0.0.1", port: int = RCON_PORT, password: str = PASSWORD):
        self.sock = socket.create_connection((host, port), timeout=900)
        self._id = 0
        self._send(3, password)
        # The server answers an auth request with an empty response packet, then the auth result.
        while True:
            rid, typ, _ = self._recv()
            if typ == 2:
                if rid == -1:
                    raise PermissionError("RCON auth failed")
                break

    def _send(self, typ: int, body: str) -> int:
        self._id += 1
        data = struct.pack("<ii", self._id, typ) + body.encode() + b"\x00\x00"
        self.sock.sendall(struct.pack("<i", len(data)) + data)
        return self._id

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("RCON closed")
            buf += chunk
        return buf

    def _recv(self):
        (size,) = struct.unpack("<i", self._recv_exact(4))
        data = self._recv_exact(size)
        rid, typ = struct.unpack("<ii", data[:8])
        return rid, typ, data[8:-2].decode(errors="replace")

    def cmd(self, command: str) -> str:
        rid = self._send(2, command)
        # Factorio sends exactly one response packet per command.
        while True:
            r, _, body = self._recv()
            if r == rid:
                return body

    _lua_unlocked = False

    def lua(self, code: str) -> str:
        """Run Lua silently (/sc). Use rcon.print(...) to return values."""
        if not Rcon._lua_unlocked:
            # The first Lua command only triggers the "disables achievements" warning,
            # so send a harmless one twice. This only affects the separate test save.
            self.cmd("/sc local _=0")
            self.cmd("/sc local _=0")
            Rcon._lua_unlocked = True
        return self.cmd("/sc " + code)

    def close(self):
        self.sock.close()


def start_server(save: Path = ROOT / "server" / "sandbox.zip") -> subprocess.Popen:
    """Start a fresh server from a throwaway copy of the pristine sandbox save."""
    import shutil
    run = ROOT / "server" / "data" / "run.zip"
    run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(save, run)
    save = run
    log = open(ROOT / "server" / "server.log", "w")
    proc = subprocess.Popen(
        [FACTORIO, "--config", str(ROOT / "server" / "config.ini"),
         "--start-server", str(save),
         "--server-settings", str(ROOT / "server" / "server-settings.json"),
         "--port", str(GAME_PORT),
         "--rcon-port", str(RCON_PORT), "--rcon-password", PASSWORD],
        stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            Rcon().close()
            return proc
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError("server exited; see harness/server/server.log")
            time.sleep(0.5)
    raise TimeoutError("server did not open RCON")


if __name__ == "__main__":
    import sys
    p = start_server()
    print("server up, pid", p.pid)
    if len(sys.argv) > 1:
        r = Rcon()
        print(r.lua(" ".join(sys.argv[1:])))
        r.close()
        p.terminate()
