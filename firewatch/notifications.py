"""Notification composition and simulation-only provider."""

from __future__ import annotations

import hashlib
import json
import re
import smtplib
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage

from .storage import record_notification_attempt


class SimulatedNotificationProvider:
    """A local provider that records an item and never contacts anyone."""

    def deliver(self, payload: dict, recipient: str) -> dict:
        del payload, recipient
        return {"status": "simulated", "response": "Simulation recorded locally; no external message was sent."}


class SMTPTestNotificationProvider:
    """Send explicitly synthetic test notices through a configured SMTP relay."""

    def __init__(self, *, host: str, port: int, username: str, password: str,
                 from_address: str, timeout: float = 15.0):
        if not all((host.strip(), username.strip(), password, from_address.strip())):
            raise ValueError("SMTP host, username, password, and sender are required.")
        self.host, self.port = host.strip(), int(port)
        self.username, self.password = username.strip(), password
        self.from_address, self.timeout = from_address.strip(), float(timeout)

    def deliver(self, payload: dict, recipient: str) -> dict:
        if not payload.get("test_only") or not payload.get("synthetic_data"):
            raise ValueError("Email provider accepts synthetic test alerts only.")
        msg = EmailMessage()
        msg["Subject"] = "[FIREWATCH SYNTHETIC TEST] Simulated forest-fire alert"
        msg["From"] = self.from_address
        msg["To"] = recipient
        msg.set_content(
            "FIREWATCH HACKATHON TEST — SYNTHETIC DATA ONLY\n\n"
            "This is a software demonstration. It is not a real fire alert and must not be used for emergency response.\n\n"
            f"Scenario record: {payload.get('observation_id', 'unknown')}\n"
            f"Synthetic event time (UTC): {payload.get('observation_datetime_utc', 'unknown')}\n"
            f"Synthetic coordinates: {payload.get('latitude')}, {payload.get('longitude')}\n"
            f"Synthetic priority: {payload.get('priority_category', 'unknown')} "
            f"(score {payload.get('priority_score', 'unknown')})\n"
            f"Synthetic scenario: {payload.get('synthetic_scenario', 'generated') }\n\n"
            "The locations, sensor-like values, and labels were generated locally. "
            "This message does not report a real-world observation.\n"
        )
        with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(self.username, self.password)
            refused = server.send_message(msg)
        if recipient in refused:
            raise RuntimeError("SMTP relay refused this recipient.")
        return {"status": "provider-accepted", "response": "SMTP relay accepted the test message; inbox delivery is unconfirmed."}


class FormSubmitNotificationProvider:
    """Submit an explicit, recent NASA FIRMS review request through FormSubmit."""

    def __init__(self, *, timeout: float = 20.0):
        self.timeout = float(timeout)

    def deliver(self, payload: dict, recipient: str) -> dict:
        if payload.get("synthetic_data") or payload.get("test_only"):
            raise ValueError("ERS email is disabled for synthetic/demo observations.")
        if "NASA FIRMS Area API" not in str(payload.get("source", "")):
            raise ValueError("ERS email is allowed only for a NASA FIRMS Area API observation.")
        try:
            acquired = datetime.fromisoformat(str(payload.get("observation_datetime_utc", "")).replace("Z", "+00:00"))
            if acquired.tzinfo is None:
                acquired = acquired.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            raise ValueError("The observation has no valid acquisition timestamp.") from None
        age_hours = (datetime.now(timezone.utc) - acquired.astimezone(timezone.utc)).total_seconds() / 3600
        if age_hours < -0.25 or age_hours > 144:
            raise ValueError("ERS email is limited to FIRMS observations from the last six days.")
        if not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}", recipient.strip()):
            raise ValueError("A valid ERS recipient email is required.")

        message = (
            "FIREWATCH REVIEW REQUEST — UNCONFIRMED SATELLITE THERMAL ANOMALY\n\n"
            "This is an informational request for manual authority review. It is not a confirmed wildfire, "
            "severity assessment, or dispatch instruction. Please verify independently and decide any response.\n\n"
            f"Observation ID: {payload.get('observation_id', 'unknown')}\n"
            f"Acquired (UTC): {payload.get('observation_datetime_utc', 'unknown')}\n"
            f"Coordinates: {payload.get('latitude')}, {payload.get('longitude')}\n"
            f"Map: {payload.get('map_link', '')}\n"
            f"Satellite / instrument / product: {payload.get('satellite', 'Unknown')} / "
            f"{payload.get('instrument', 'Unknown')} / {payload.get('product', 'Unknown')}\n"
            f"FIRMS confidence category: {payload.get('confidence', 'Unknown')}\n"
            f"Fire radiative power (MW): {payload.get('frp_mw', 'Unknown')}\n"
            f"FireWatch review priority (not probability): {payload.get('priority_category', 'Unknown')} "
            f"({payload.get('priority_score', 'Unknown')})\n"
            f"Jurisdiction: {payload.get('jurisdiction', 'Unknown — manual review required')}\n"
            f"Source: {payload.get('source', 'NASA FIRMS Area API')}\n\n"
            f"Requested by FireWatch: {payload.get('trigger_reason', 'Manual review request')}\n"
            "No action has been taken by FireWatch. Confirm or reject this observation through your normal process.\n"
        )
        form_data = {
            "_subject": "[FireWatch] Satellite thermal anomaly — manual review requested",
            "_template": "table",
            "name": "FireWatch AI review queue",
            "observation_id": str(payload.get("observation_id", "unknown")),
            "acquisition_time_utc": str(payload.get("observation_datetime_utc", "unknown")),
            "coordinates": f"{payload.get('latitude')}, {payload.get('longitude')}",
            "map_link": str(payload.get("map_link", "")),
            "satellite_instrument": f"{payload.get('satellite', 'Unknown')} / {payload.get('instrument', 'Unknown')}",
            "confidence": str(payload.get("confidence", "Unknown")),
            "frp_mw": str(payload.get("frp_mw", "Unknown")),
            "review_priority": f"{payload.get('priority_category', 'Unknown')} ({payload.get('priority_score', 'Unknown')})",
            "jurisdiction": str(payload.get("jurisdiction", "Unknown — manual review required")),
            "message": message,
        }
        endpoint = "https://formsubmit.co/ajax/" + recipient.strip()
        request = urllib.request.Request(endpoint,
            data=json.dumps(form_data, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read(64 * 1024).decode("utf-8", errors="replace")
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(f"FormSubmit returned HTTP {response.status}.")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"FormSubmit returned HTTP {exc.code}; check recipient activation and service status.") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RuntimeError("Could not reach FormSubmit; no delivery was confirmed.") from None
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            raise RuntimeError("FormSubmit returned an unrecognized response; delivery is unconfirmed.") from None
        if str(result.get("success", "")).lower() not in {"true", "1"}:
            detail = str(result.get("message", "Submission was not accepted."))[:240]
            raise RuntimeError(f"FormSubmit did not accept the request: {detail}")
        return {"status": "provider-accepted",
                "response": "FormSubmit accepted the review request; inbox delivery and recipient activation are unconfirmed."}


def build_notification_payload(observation: dict, *, trigger_reason: str,
                               routing_reason: str, recipient: str) -> dict:
    lat, lon = float(observation["latitude"]), float(observation["longitude"])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Notification coordinates are invalid.")
    return {
        "alert_id": str(uuid.uuid4()),
        "observation_id": str(observation["observation_id"]),
        "observation_datetime_utc": str(observation.get("acq_datetime", "")),
        "latitude": lat,
        "longitude": lon,
        "map_link": f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}",
        "satellite": str(observation.get("satellite") or "Unknown"),
        "instrument": str(observation.get("instrument") or "Unknown"),
        "product": str(observation.get("product") or "Unknown"),
        "source": str(observation.get("source_name") or "Unknown"),
        "synthetic_data": "SYNTHETIC" in str(observation.get("source_name") or "").upper(),
        "test_only": "SYNTHETIC" in str(observation.get("source_name") or "").upper(),
        "confidence": observation.get("confidence"),
        "frp_mw": observation.get("frp"),
        "bright_ti4": observation.get("bright_ti4"),
        "bright_ti5": observation.get("bright_ti5"),
        "priority_score": observation.get("priority_score"),
        "priority_category": observation.get("priority_category"),
        "review_status": observation.get("review_status", "Awaiting review"),
        "explanation": str(observation.get("explanation") or "No score explanation available."),
        "data_quality_warnings": str(observation.get("missing_factors") or ""),
        "recipient": recipient,
        "trigger_reason": trigger_reason,
        "routing_reason": routing_reason,
        "jurisdiction": observation.get("verified_jurisdiction") or "Jurisdiction unknown — manual review required.",
        "verification_notice": "Unverified satellite thermal anomaly; not a confirmed wildfire.",
        "recommended_verification": [
            "Review the source product and confidence metadata.",
            "Check coincident imagery and recent independent observations.",
            "Verify the location with a qualified human reviewer before action.",
        ],
    }


def notification_dedupe_key(observation_id: str, trigger_reason: str, recipient: str | None = None) -> str:
    basis = f"{observation_id.strip()}|{trigger_reason.strip().casefold()}|{(recipient or '').strip().casefold()}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def payload_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def dispatch_notification(conn, notification_id: str, provider, *, max_attempts: int = 2) -> dict:
    """Attempt delivery through the explicitly selected provider and audit each result.

    The dashboard selects only ``SimulatedNotificationProvider``. Other provider
    statuses are recorded verbatim; this function never upgrades a status to
    delivered on the provider's behalf.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    row = conn.execute("SELECT payload_json, recipient, status FROM notifications WHERE notification_id=?",
                       (notification_id,)).fetchone()
    if row is None:
        raise ValueError("Unknown notification id")
    if row["status"] not in {"queued", "failed"}:
        return {"status": row["status"], "attempts": 0, "response": "Notification is not pending."}
    payload, recipient = json.loads(row["payload_json"]), row["recipient"]
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            result = provider.deliver(payload, recipient)
            status = result.get("status")
            if status not in {"simulated", "sent", "provider-accepted", "delivered"}:
                raise ValueError("Provider returned an unsupported delivery status")
            response = str(result.get("response", ""))
            record_notification_attempt(conn, notification_id, status=status, details=response,
                                        retry_count=attempt - 1)
            return {"status": status, "attempts": attempt, "response": response}
        except Exception as exc:  # provider failure becomes an auditable state
            last_error = f"{type(exc).__name__}: {exc}"
            record_notification_attempt(conn, notification_id, status="failed", details=last_error,
                                        retry_count=attempt - 1, error_details=last_error)
            if attempt < max_attempts:
                record_notification_attempt(conn, notification_id, status="queued",
                                            details=f"Retry scheduled after attempt {attempt}.",
                                            retry_count=attempt)
    return {"status": "failed", "attempts": max_attempts, "response": last_error}
