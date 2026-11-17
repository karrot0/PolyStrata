"""setup.download over several ranged connections: whole file, resume from a single-connection .part and from the
parallel progress file, and a server without ranges.  Mocked network - nothing is downloaded."""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

BODY = bytes(range(256)) * 40                          # 10,240 bytes
PIECE = 1000                                           # 11 pieces


class Response(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Server:
    """Answers HEAD with the length and GET with the asked range (or the whole body when ranges=False)."""

    def __init__(self, ranges=True):
        self.ranges = ranges
        self.asked = []
        self.lock = threading.Lock()

    def urlopen(self, req, timeout=None):
        h = {k.lower(): v for k, v in req.header_items()}
        if req.get_method() == "HEAD":
            return Response(b"", 200, {"Content-Length": str(len(BODY))})
        rng = h.get("range")
        with self.lock:
            self.asked.append(rng)
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng or "")
        if not self.ranges or not m:
            return Response(BODY, 200, {"Content-Length": str(len(BODY))})
        lo = int(m.group(1))
        hi = int(m.group(2)) + 1 if m.group(2) else len(BODY)
        return Response(BODY[lo:hi], 206, {"Content-Range": f"bytes {lo}-{hi - 1}/{len(BODY)}"})


def run(server, dst, conns="4"):
    out = io.StringIO()
    with mock.patch.object(setup.urllib.request, "urlopen", server.urlopen), \
            mock.patch.object(setup, "DL_PIECE", PIECE), \
            mock.patch.object(setup.time, "sleep", lambda s: None), \
            mock.patch.dict(setup.os.environ, {"STRATA_DOWNLOAD_CONNECTIONS": conns, "STRATA_SOURCE": "huggingface"}), \
            contextlib.redirect_stdout(out):
        setup.download("https://example.com/m.gguf", dst)
    return out.getvalue()


class ParallelDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dst = Path(self.tmp.name) / "m.gguf"

    def tearDown(self):
        self.tmp.cleanup()

    def leftovers(self):
        return sorted(p.name for p in self.dst.parent.iterdir() if p.name not in ("m.gguf", "m.gguf.done"))

    def test_whole_file_in_pieces(self):
        s = Server()
        out = run(s, self.dst)
        self.assertEqual(self.dst.read_bytes(), BODY)
        self.assertTrue(setup.done(self.dst))
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(len(s.asked), 11)
        self.assertIn("4 connections", out)

    def test_takes_over_a_single_connection_part(self):
        self.dst.with_name("m.gguf.part").write_bytes(BODY[:4321])
        s = Server()
        run(s, self.dst)
        self.assertEqual(self.dst.read_bytes(), BODY)
        self.assertEqual(min(int(r.split("=")[1].split("-")[0]) for r in s.asked), 4321)

    def test_resumes_from_its_progress_file(self):
        ppart = self.dst.with_name("m.gguf.ppart")
        ppart.write_bytes(BODY[:3000] + b"\0" * 1000 + BODY[4000:5000])   # pieces 0, 1, 2 and 4 written
        setup.parallel_state(self.dst).write_text(json.dumps(
            {"total": len(BODY), "piece": PIECE, "start": 0, "done": [0, 1, 2, 4]}))
        s = Server()
        run(s, self.dst)
        self.assertEqual(self.dst.read_bytes(), BODY)
        self.assertNotIn("bytes=0-999", s.asked)
        self.assertNotIn("bytes=4000-4999", s.asked)
        self.assertIn("bytes=3000-3999", s.asked)
        self.assertEqual(self.leftovers(), [])

    def test_a_server_without_ranges_gets_one_connection(self):
        s = Server(ranges=False)
        out = run(s, self.dst)
        self.assertEqual(self.dst.read_bytes(), BODY)
        self.assertIn("over one connection", out)
        self.assertEqual(self.leftovers(), [])

    def test_one_connection_is_the_old_path(self):
        s = Server()
        run(s, self.dst, conns="1")
        self.assertEqual(self.dst.read_bytes(), BODY)
        self.assertEqual(s.asked, ["bytes=0-"])

    def test_small_files_stay_on_one_connection(self):
        s = Server()
        with mock.patch.object(setup, "DL_PIECE", 1 << 20):
            out = io.StringIO()
            with mock.patch.object(setup.urllib.request, "urlopen", s.urlopen), contextlib.redirect_stdout(out), \
                    mock.patch.dict(setup.os.environ, {"STRATA_SOURCE": "huggingface"}):
                setup.download("https://example.com/m.gguf", self.dst)
        self.assertEqual(s.asked, ["bytes=0-"])


if __name__ == "__main__":
    unittest.main()
