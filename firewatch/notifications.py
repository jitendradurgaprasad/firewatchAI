"""Notification composition and simulation-only provider."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

from .storage import record_notification_attempt


class SimulatedNotificationProvider:
    """A local provider that records an item and never contacts anyone."""

    def deliver(self, payload: dict, recipient: str) -> dict:
        del payload, recipient
        return {"status": "simulated", "response": "Simulation recorded locally; no external message was sent."}


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


def notification_dedupe_key(observation_id: str, trigger_reason: str) -> str:
    basis = f"{observation_id.strip()}|{trigger_reason.strip().casefold()}"
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
