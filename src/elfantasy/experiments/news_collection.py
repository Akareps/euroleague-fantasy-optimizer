"""Archive reviewed public excerpts, optional HTML, and timestamped transcripts.

Discovery and multilingual attribution are reviewed by a person/assistant. The
rule extractor suggests candidates; it cannot approve its own predictions.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import yaml
from bs4 import BeautifulSoup

from elfantasy.data.http import USER_AGENT, HttpClient
from elfantasy.experiments.history import utc
from elfantasy.experiments.news_minutes import EXPERIMENT, classify, normalise, sentiment


def transcript_segments(text):
    """Read SRT/VTT captions without losing the original speaker/video offsets."""
    segments = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = block.splitlines()
        timing = next((i for i, line in enumerate(lines) if " --> " in line), None)
        if timing is None:
            continue
        start, end = lines[timing].split(" --> ", 1)
        if not re.fullmatch(r"(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3}", start.strip()):
            continue
        words = BeautifulSoup(" ".join(lines[timing + 1 :]), "html.parser").get_text(
            " ", strip=True
        )
        if words:
            segments.append({"start": start.strip(), "end": end.split()[0], "text": words})
    return segments


def candidates(text, source, scope):
    """Named excerpts are review candidates, never automatically actionable."""
    segments = transcript_segments(text) if source.get("format") in {"srt", "vtt"} else []
    if not segments:
        segments = [
            {"text": part.strip()} for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()
        ]
    rows = []
    for segment in segments:
        excerpt = segment["text"]
        if len(excerpt) > 1000:
            continue
        folded = " " + normalise(excerpt) + " "
        matched = [
            p
            for p in scope
            if p["club"] == source.get("club")
            and any(
                " " + normalise(alias) + " " in folded
                for alias in [p["player"], *p.get("aliases", [])]
            )
        ]
        if not matched:
            continue
        extracted = classify(excerpt)
        for player in matched:
            rows.append(
                {
                    "source_id": source["id"],
                    "player": player["player"],
                    "club": player["club"],
                    "excerpt": excerpt[:600],
                    "transcript_start": segment.get("start"),
                    "transcript_end": segment.get("end"),
                    "extracted": extracted,
                    "sentiment": sentiment(excerpt),
                    "reviewed": False,
                    "ambiguous_sentence": len(matched) > 1,
                }
            )
    return rows


def collect(manifest_path: Path, out: Path, *, fetch_pages=False):
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != EXPERIMENT or manifest.get("schema_version") != 1:
        raise ValueError("invalid news research manifest")
    out.mkdir(parents=True, exist_ok=True)
    http = HttpClient(out / "http", retries=1, timeout=12, min_interval=1.0)
    sources, claims, proposals = [], [], []
    robots = {}
    for item in manifest.get("sources", []):
        source = {k: v for k, v in item.items() if k != "claims"}
        now = datetime.now(timezone.utc).isoformat()
        source.update(retrieved_at=now, retrieval_status="reviewed_excerpts_only")
        body, text = None, None
        if item.get("transcript_path"):
            path = (manifest_path.parent / item["transcript_path"]).resolve()
            # Local captions are deliberately imported, never discovered across the PC.
            body = path.read_text(encoding="utf-8-sig")
            text = body
            source["retrieval_status"] = "local_transcript"
        elif fetch_pages:
            url = item["url"]
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc or parsed.username:
                raise ValueError("source URLs must be public HTTPS URLs")
            origin = f"https://{parsed.netloc}"
            if origin not in robots:
                robot_text = http.get(origin + "/robots.txt", use_cache=False)
                parser = RobotFileParser()
                if robot_text is not None:
                    parser.parse(robot_text.splitlines())
                robots[origin] = parser if robot_text is not None else None
            parser = robots[origin]
            source["robots_checked"] = parser is not None
            if parser is None:
                source["retrieval_status"] = "robots_unavailable; kept reviewed excerpts"
            elif not parser.can_fetch(USER_AGENT, url):
                source["retrieval_status"] = "robots_disallowed; kept reviewed excerpts"
            else:
                body = http.get(url, use_cache=False)
                if body is None:
                    source["retrieval_status"] = "page_unavailable; kept reviewed excerpts"
                else:
                    soup = BeautifulSoup(body, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    text = (soup.find("article") or soup.find("main") or soup).get_text(
                        " ", strip=True
                    )
                    source["retrieval_status"] = "html_snapshot"
        if body is not None:
            digest = sha256(body.encode("utf-8")).hexdigest()
            path = out / "sources" / (digest + ".txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            source.update(content_sha256=digest, local_snapshot=str(path.relative_to(out)))
            proposals.extend(candidates(text, source, manifest.get("scope", [])))
        for row in item.get("claims", []):
            reviewed = item.get("reviewed_at")
            if reviewed:
                utc(reviewed)
            verified = bool(
                row.get("reviewed") and reviewed and row.get("verification") == "manual_web_review"
            )
            method = "manual_web_review" if verified else "unverified"
            if text is not None and normalise(row["excerpt"]) in normalise(text):
                verified, method = True, "literal_excerpt_in_snapshot"
            claim = {
                **row,
                "round": manifest["round"],
                "source_id": item["id"],
                "source_url": item["url"],
                "source_type": row.get("source_type", item["source_type"]),
                "published_at": item["published_at"],
                "publication_precision": item.get("publication_precision", "second"),
                "observed_at": reviewed
                if method == "manual_web_review"
                else source["retrieved_at"],
                "excerpt_verified": verified,
                "verification_method": method,
                "language": item.get("language", "en"),
            }
            claims.append(claim)
        sources.append(source)
    captured = datetime.now(timezone.utc).isoformat()
    snapshot = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "round": manifest["round"],
        "captured_at": captured,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "collection_mode": "reviewed excerpts plus optional public HTML/local captions",
        "scope": manifest.get("scope", []),
        "sources": sources,
        "claims": claims,
        "candidates": proposals,
        "limitations": manifest.get("limitations", []),
    }
    (out / "snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return snapshot
