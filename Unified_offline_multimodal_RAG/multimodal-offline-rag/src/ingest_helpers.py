from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


def concept_display(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    if value.get("text") or value.get("display") or value.get("value"):
        return str(value.get("text") or value.get("display") or value.get("value"))
    codings = value.get("coding", [])
    for coding in codings:
        if isinstance(coding, dict) and (coding.get("display") or coding.get("code")):
            return str(coding.get("display") or coding.get("code"))
    return ""


def _resources(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and data.get("resourceType") == "Bundle":
        return [entry["resource"] for entry in data.get("entry", []) if isinstance(entry, dict) and isinstance(entry.get("resource"), dict)]
    if isinstance(data, dict) and data.get("resourceType"):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _human_name(raw: Any) -> str:
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if not isinstance(raw, dict):
        return "Not provided"
    given = raw.get("given", [])
    return " ".join([*(str(x) for x in given), str(raw.get("family", ""))]).strip() or "Not provided"


def _observation(resource: dict[str, Any]) -> str:
    label = concept_display(resource.get("code", {})) or "Observation"
    value = ""
    if isinstance(resource.get("valueQuantity"), dict):
        quantity = resource["valueQuantity"]
        value = f"{quantity.get('value', '')} {quantity.get('unit') or quantity.get('code') or ''}".strip()
    elif "valueString" in resource:
        value = str(resource.get("valueString", ""))
    elif "valueCodeableConcept" in resource:
        value = concept_display(resource["valueCodeableConcept"])
    elif "valueBoolean" in resource:
        value = str(resource["valueBoolean"])
    date = resource.get("effectiveDateTime") or resource.get("issued") or ""
    return f"{label}: {value}" + (f" (date: {date})" if date else "")


def extract_fhir_json(file_path: str) -> str:
    """Convert common FHIR JSON resources and Synthea patient Bundles to searchable text."""
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    resources = _resources(data)
    if not resources:
        raise ValueError("JSON did not contain FHIR resources or a Bundle.")

    sections: dict[str, list[str]] = {
        "PATIENT INFORMATION": [], "MEDICAL CONDITIONS / DIAGNOSES": [],
        "MEDICATIONS": [], "ALLERGIES": [], "LAB RESULTS / OBSERVATIONS": [],
        "PROCEDURES": [], "ENCOUNTERS": [], "DIAGNOSTIC REPORTS": [],
        "IMMUNIZATIONS": [], "CARE PLANS": [], "CLINICAL NOTES": [],
    }
    for resource in resources:
        kind = resource.get("resourceType", "")
        if kind == "Patient":
            sections["PATIENT INFORMATION"].extend([
                f"Patient ID: {resource.get('id', 'Not provided')}",
                f"Name: {_human_name(resource.get('name', []))}",
                f"Gender: {resource.get('gender', 'Not provided')}",
                f"Date of birth: {resource.get('birthDate', 'Not provided')}",
            ])
        elif kind == "Condition":
            value = concept_display(resource.get("code", {}))
            if value:
                sections["MEDICAL CONDITIONS / DIAGNOSES"].append(value)
        elif kind in {"MedicationRequest", "MedicationStatement", "Medication", "MedicationAdministration"}:
            value = concept_display(resource.get("medicationCodeableConcept", {}))
            if not value:
                value = concept_display(resource.get("medicationReference", {})) or concept_display(resource.get("code", {}))
            if value:
                status = resource.get("status", "")
                sections["MEDICATIONS"].append(value + (f" (status: {status})" if status else ""))
        elif kind == "AllergyIntolerance":
            value = concept_display(resource.get("code", {})) or "Allergy entry"
            status = resource.get("clinicalStatus", {})
            display = concept_display(status)
            sections["ALLERGIES"].append(value + (f" (status: {display})" if display else ""))
        elif kind == "Observation":
            value = _observation(resource)
            if value.strip(": "):
                sections["LAB RESULTS / OBSERVATIONS"].append(value)
        elif kind == "Procedure":
            value = concept_display(resource.get("code", {}))
            if value:
                sections["PROCEDURES"].append(value)
        elif kind == "Encounter":
            types = resource.get("type", [])
            value = "; ".join(filter(None, (concept_display(item) for item in types)))
            value = value or concept_display(resource.get("class", {}))
            if value:
                date = resource.get("period", {}).get("start", "")
                sections["ENCOUNTERS"].append(value + (f" (date: {date})" if date else ""))
        elif kind == "DiagnosticReport":
            report = concept_display(resource.get("code", {}))
            conclusion = resource.get("conclusion", "")
            if report or conclusion:
                sections["DIAGNOSTIC REPORTS"].append(": ".join(x for x in (report, conclusion) if x))
        elif kind == "Immunization":
            vaccine = concept_display(resource.get("vaccineCode", {}))
            if vaccine:
                sections["IMMUNIZATIONS"].append(vaccine)
        elif kind == "CarePlan":
            title = resource.get("title", "") or concept_display(resource.get("category", [{}])[0] if resource.get("category") else {})
            if title:
                sections["CARE PLANS"].append(title)
        elif kind == "DocumentReference":
            for item in resource.get("content", []):
                attachment = item.get("attachment", {}) if isinstance(item, dict) else {}
                if attachment.get("data"):
                    try:
                        text = base64.b64decode(attachment["data"]).decode("utf-8", errors="ignore").strip()
                        if text:
                            sections["CLINICAL NOTES"].append(text)
                    except (ValueError, TypeError):
                        pass
                elif attachment.get("title"):
                    sections["CLINICAL NOTES"].append(str(attachment["title"]))

    lines = ["PATIENT MEDICAL RECORD", "Source format: FHIR JSON", ""]
    for heading, values in sections.items():
        if values:
            lines.extend([heading, *[f"- {value}" for value in values], ""])
    return "\n".join(lines).strip()
