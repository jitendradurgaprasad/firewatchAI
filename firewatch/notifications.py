"""Notification composition and simulation-only provider."""

from __future__ import annotations

import hashlib
import json
import smtplib
import uuid
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
