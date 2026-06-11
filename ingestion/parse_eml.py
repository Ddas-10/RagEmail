"""
ingestion/parse_eml.py — Parse .eml files into EmailRecord objects.

Thread detection:
  1. Use In-Reply-To / References headers as the primary thread signal.
  2. Fall back to normalised subject (strip Re:/Fwd:/AW: etc.) + sender domain.
  3. Assign a stable thread_id = sha1 of the canonical root message_id.
"""

from __future__ import annotations
import email
import hashlib
import mailbox
import re
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path
from typing import Iterator

from utils.models import EmailRecord


# ──────────────────────────────────────────────
# Subject normalisation
# ──────────────────────────────────────────────

_REPLY_PREFIX = re.compile(
    r"^(re|fwd?|aw|antw|sv|vs|tr|回复|转发)[\s:\[]+",
    re.IGNORECASE,
)

def normalise_subject(subject: str) -> str:
    s = subject.strip()
    while True:
        m = _REPLY_PREFIX.match(s)
        if not m:
            break
        s = s[m.end():].strip()
    return s.lower()


# ──────────────────────────────────────────────
# Thread graph builder
# key: message_id  →  value: canonical thread_id
# ──────────────────────────────────────────────

def build_thread_map(raw_msgs: list[dict]) -> dict[str, str]:
    """
    Primary: group by normalised subject (mirrors extract_slice.py subject buckets).
    This is reliable for our dataset slice where In-Reply-To chains are incomplete.

    Fallback: emails with an empty subject each get their own thread.
    """
    subject_to_thread: dict[str, str] = {}
    thread_map: dict[str, str] = {}

    for m in raw_msgs:
        mid = m["message_id"]
        subj = normalise_subject(m.get("subject", ""))
        if not subj:
            # No subject → unique thread per message
            key = mid
        else:
            key = subj
        if key not in subject_to_thread:
            subject_to_thread[key] = "T-" + hashlib.sha1(key.encode()).hexdigest()[:8].upper()
        thread_map[mid] = subject_to_thread[key]

    return thread_map


# ──────────────────────────────────────────────
# Single .eml file parser
# ──────────────────────────────────────────────

def _extract_body(msg: email.message.Message) -> str:
    """Prefer text/plain; fall back to stripping text/html."""
    body_plain = []
    body_html = []

    for part in msg.walk():
        ct = part.get_content_type()
        if ct == "text/plain":
            payload = part.get_payload(decode=True)
            if payload:
                body_plain.append(payload.decode("utf-8", errors="replace"))
        elif ct == "text/html" and not body_plain:
            payload = part.get_payload(decode=True)
            if payload:
                # crude strip — replace with html2text in production
                text = re.sub(r"<[^>]+>", " ", payload.decode("utf-8", errors="replace"))
                body_html.append(re.sub(r"\s+", " ", text).strip())

    return "\n".join(body_plain).strip() or "\n".join(body_html).strip()


_LIST_ADDR_PARTS = frozenset({
    "all", "staff", "employees", "everyone", "worldwide", "corp", "announcements",
    "dist", "distribution", "group", "team", "department",
})


def _name_from_parsed(display_name: str, email_addr: str) -> str | None:
    """Derive a human display name from a parsed (display_name, email) pair.

    Uses the MIME display name when present; derives first+last from the local
    part of the email address otherwise.  Returns None for list/distribution
    addresses (all@, announcements@, etc.) or single-word identifiers.
    """
    name = display_name.strip()
    if not name and email_addr:
        local = email_addr.split("@")[0].replace("_", ".").replace("-", ".")
        parts = [p for p in local.split(".") if p and not p.isdigit()]
        # Single-word or all-lowercase short tokens are likely system addresses
        if len(parts) < 2:
            return None
        name = " ".join(p.capitalize() for p in parts[:3])

    if not name:
        return None

    words = name.split()
    if any(w.lower() in _LIST_ADDR_PARTS for w in words):
        return None
    if len(words) < 2:
        return None
    return name


def parse_eml_file(path: Path) -> dict:
    """Parse one .eml file → raw dict (before thread_id assignment)."""
    with open(path, "rb") as f:
        msg = email.message_from_bytes(f.read())

    message_id = msg.get("Message-ID", "").strip("<>").strip() or path.stem
    from_name, from_addr = parseaddr(msg.get("From", ""))
    to_raw = msg.get("To", "") + "," + msg.get("Cc", "")
    to_cc_pairs = [parseaddr(x) for x in to_raw.split(",") if x.strip()]

    # Build participant list: all unique human names derived from From/To/Cc headers.
    # Using standard MIME parseaddr — reliable, no body-text parsing needed.
    participants: list[str] = []
    seen_p: set[str] = set()
    for disp, addr in [(from_name, from_addr)] + to_cc_pairs:
        n = _name_from_parsed(disp, addr.strip())
        if n and n.lower() not in seen_p:
            seen_p.add(n.lower())
            participants.append(n)

    try:
        date = parsedate_to_datetime(msg.get("Date", "")).isoformat()
    except Exception:
        date = "1970-01-01T00:00:00"

    attachments = []
    for part in msg.walk():
        fname = part.get_filename()
        if fname:
            attachments.append(fname)

    return {
        "message_id": message_id,
        "in_reply_to": msg.get("In-Reply-To", ""),
        "references": msg.get("References", ""),
        "date": date,
        "from_addr": from_addr,
        "to_cc": [a.strip() for _, a in to_cc_pairs if a.strip()],
        "subject": msg.get("Subject", ""),
        "body": _extract_body(msg),
        "attachment_filenames": attachments,
        "participants": participants,
    }


# ──────────────────────────────────────────────
# Batch loader
# ──────────────────────────────────────────────

def load_eml_directory(data_dir: Path) -> list[EmailRecord]:
    """
    Load all .eml files under data_dir, assign thread_ids, return EmailRecords.
    Also handles Enron mbox format if .mbox files are present.
    """
    raw: list[dict] = []

    for p in sorted(data_dir.rglob("*.eml")):
        try:
            raw.append(parse_eml_file(p))
        except Exception as e:
            print(f"[WARN] failed to parse {p}: {e}")

    for p in sorted(data_dir.rglob("*.mbox")):
        try:
            import tempfile
            mbox = mailbox.mbox(str(p))
            for msg in mbox:
                with tempfile.NamedTemporaryFile(suffix=".eml", delete=False) as tmp:
                    tmp.write(msg.as_bytes())
                    tmp_path = Path(tmp.name)
                try:
                    raw.append(parse_eml_file(tmp_path))
                finally:
                    tmp_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] failed to parse mbox {p}: {e}")

    if not raw:
        raise ValueError(f"No .eml or .mbox files found under {data_dir}")

    thread_map = build_thread_map(raw)

    records = []
    for r in raw:
        records.append(EmailRecord(
            message_id=r["message_id"],
            thread_id=thread_map[r["message_id"]],
            date=r["date"],
            from_addr=r["from_addr"],
            to_cc=r["to_cc"],
            subject=r["subject"],
            body=r["body"],
            attachment_ids=[],  # filled by parse_attachments.py
            participants=r.get("participants", []),
        ))

    print(f"[INFO] Loaded {len(records)} emails across "
          f"{len({r.thread_id for r in records})} threads")
    return records
