import os
import glob
import json
import base64
import click

# Import our custom modules from the src/ directory
from src.document_loaders import get_loader
from src.semantic_processor import SemanticProcessor


SOURCE_DIR = "source_documents"
INDEX_DIR = "faissindex"


def ensure_directories():
    """Ensure necessary directories exist before starting."""
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)


# ============================================================
# FHIR JSON EXTRACTION
# ============================================================

def get_human_name(name_data):
    """Convert FHIR HumanName object into a readable name."""
    
    if not name_data:
        return "Unknown"

    if isinstance(name_data, list):
        name_data = name_data[0]

    given = " ".join(name_data.get("given", []))
    family = name_data.get("family", "")

    return f"{given} {family}".strip()


def get_reference_display(resource):
    """
    Extract readable text from common FHIR fields.
    """

    if not resource:
        return ""

    if isinstance(resource, str):
        return resource

    if isinstance(resource, dict):
        return (
            resource.get("text")
            or resource.get("display")
            or resource.get("value")
            or ""
        )

    return ""


def extract_fhir_json(file_path):
    """
    Extract useful clinical information from a FHIR JSON file
    and convert it into a readable medical record.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    lines.append("PATIENT MEDICAL RECORD")
    lines.append("=" * 50)

    # --------------------------------------------------------
    # Get resources from a FHIR Bundle
    # --------------------------------------------------------

    resources = []

    if isinstance(data, dict):

        # Standard FHIR Bundle
        if data.get("resourceType") == "Bundle":

            for entry in data.get("entry", []):

                if "resource" in entry:
                    resources.append(entry["resource"])

        # Single FHIR resource
        elif "resourceType" in data:
            resources.append(data)

        # Generic JSON fallback
        else:
            resources.append(data)

    elif isinstance(data, list):
        resources = data

    # --------------------------------------------------------
    # Process resources
    # --------------------------------------------------------

    patient_found = False
    conditions = []
    medications = []
    observations = []
    procedures = []
    encounters = []
    diagnostic_reports = []
    clinical_notes = []

    for resource in resources:

        if not isinstance(resource, dict):
            continue

        resource_type = resource.get("resourceType", "")

        # ====================================================
        # PATIENT
        # ====================================================

        if resource_type == "Patient":

            patient_found = True

            name = get_human_name(resource.get("name", []))
            gender = resource.get("gender", "Unknown")
            birth_date = resource.get("birthDate", "Unknown")

            patient_id = resource.get("id", "Unknown")

            lines.append("")
            lines.append("PATIENT INFORMATION")
            lines.append(f"Patient ID: {patient_id}")
            lines.append(f"Name: {name}")
            lines.append(f"Gender: {gender}")
            lines.append(f"Date of Birth: {birth_date}")

        # ====================================================
        # CONDITIONS / DIAGNOSES
        # ====================================================

        elif resource_type == "Condition":

            condition = get_reference_display(
                resource.get("code", {})
            )

            if condition:
                conditions.append(condition)

        # ====================================================
        # MEDICATIONS
        # ====================================================

        elif resource_type in [
            "MedicationRequest",
            "MedicationStatement",
            "Medication"
        ]:

            medication = ""

            # MedicationRequest often uses medicationCodeableConcept
            if "medicationCodeableConcept" in resource:
                medication = get_reference_display(
                    resource["medicationCodeableConcept"]
                )

            # Some datasets use medicationReference
            elif "medicationReference" in resource:
                medication = get_reference_display(
                    resource["medicationReference"]
                )

            elif "code" in resource:
                medication = get_reference_display(
                    resource["code"]
                )

            if medication:
                medications.append(medication)

        # ====================================================
        # LABS / OBSERVATIONS
        # ====================================================

        elif resource_type == "Observation":

            test_name = get_reference_display(
                resource.get("code", {})
            )

            value = ""

            if "valueQuantity" in resource:

                quantity = resource["valueQuantity"]

                value_number = quantity.get("value", "")
                unit = quantity.get("unit", "")

                value = f"{value_number} {unit}".strip()

            elif "valueString" in resource:

                value = resource.get("valueString", "")

            elif "valueCodeableConcept" in resource:

                value = get_reference_display(
                    resource.get("valueCodeableConcept", {})
                )

            if test_name or value:

                observation_text = (
                    f"{test_name}: {value}"
                    if test_name else value
                )

                observations.append(observation_text)

        # ====================================================
        # PROCEDURES
        # ====================================================

        elif resource_type == "Procedure":

            procedure = get_reference_display(
                resource.get("code", {})
            )

            if procedure:
                procedures.append(procedure)

        # ====================================================
        # ENCOUNTERS
        # ====================================================

        elif resource_type == "Encounter":

            encounter_type = get_reference_display(
                resource.get("type", [{}])[0]
                if resource.get("type")
                else {}
            )

            if encounter_type:
                encounters.append(encounter_type)

        # ====================================================
        # DIAGNOSTIC REPORTS
        # ====================================================

        elif resource_type == "DiagnosticReport":

            report = get_reference_display(
                resource.get("code", {})
            )

            conclusion = resource.get("conclusion", "")

            if report or conclusion:

                diagnostic_reports.append(
                    f"{report}: {conclusion}".strip(": ")
                )

        # ====================================================
        # CLINICAL NOTES / DOCUMENT REFERENCES
        # ====================================================

        elif resource_type == "DocumentReference":

            for content in resource.get("content", []):

                attachment = content.get("attachment", {})

                # Direct text
                if attachment.get("data"):

                    try:

                        decoded = base64.b64decode(
                            attachment["data"]
                        ).decode(
                            "utf-8",
                            errors="ignore"
                        )

                        if decoded.strip():
                            clinical_notes.append(decoded)

                    except Exception:
                        pass

                # URL / title fallback
                elif attachment.get("title"):
                    clinical_notes.append(
                        attachment.get("title")
                    )

    # ========================================================
    # ADD EXTRACTED SECTIONS
    # ========================================================

    if not patient_found:
        lines.append("")
        lines.append("PATIENT INFORMATION")
        lines.append(
            "Patient information could not be explicitly extracted."
        )

    if conditions:

        lines.append("")
        lines.append("MEDICAL CONDITIONS / DIAGNOSES")

        for condition in conditions:
            lines.append(f"- {condition}")

    if medications:

        lines.append("")
        lines.append("MEDICATIONS")

        for medication in medications:
            lines.append(f"- {medication}")

    if observations:

        lines.append("")
        lines.append("LAB RESULTS / OBSERVATIONS")

        for observation in observations:
            lines.append(f"- {observation}")

    if procedures:

        lines.append("")
        lines.append("PROCEDURES")

        for procedure in procedures:
            lines.append(f"- {procedure}")

    if encounters:

        lines.append("")
        lines.append("ENCOUNTERS")

        for encounter in encounters:
            lines.append(f"- {encounter}")

    if diagnostic_reports:

        lines.append("")
        lines.append("DIAGNOSTIC REPORTS")

        for report in diagnostic_reports:
            lines.append(f"- {report}")

    if clinical_notes:

        lines.append("")
        lines.append("CLINICAL NOTES")

        for note in clinical_notes:
            lines.append(note)

    return "\n".join(lines)


# ============================================================
# MAIN INGESTION PIPELINE
# ============================================================

@click.command()
def main():
    """
    Multimodal Data Ingestion Pipeline.

    Supports:
    - PDF
    - DOCX
    - Images
    - Audio
    - FHIR JSON medical records
    """

    ensure_directories()

    print(f"Starting ingestion from: {SOURCE_DIR}/")

    # Grab all files in the source directory
    all_files = glob.glob(
        os.path.join(SOURCE_DIR, "*.*")
    )

    if not all_files:

        print(
            f"No files found in {SOURCE_DIR}/."
        )

        return

    text_corpus = []
    error_log = []

    # ========================================================
    # PHASE 1: EXTRACTION
    # ========================================================

    print("\n--- Phase 1: File Extraction ---")

    for file_path in all_files:

        filename = os.path.basename(file_path)

        print(f"Processing: [{filename}]")

        try:

            # ------------------------------------------------
            # JSON / FHIR FILE
            # ------------------------------------------------

            if filename.lower().endswith(".json"):

                full_text = extract_fhir_json(file_path)

            # ------------------------------------------------
            # EXISTING MULTIMODAL FILE LOADER
            # ------------------------------------------------

            else:

                loader = get_loader(file_path)

                extracted_text_list = loader.load(
                    file_path
                )

                full_text = "\n".join(
                    extracted_text_list
                )

            # ------------------------------------------------
            # STORE EXTRACTED DATA
            # ------------------------------------------------

            if full_text.strip():

                text_corpus.append({
                    "text": full_text,
                    "source": filename
                })

                print(
                    f" -> Successfully extracted text from {filename}"
                )

            else:

                print(
                    f" -> Warning: No readable text found in {filename}"
                )

        except ValueError as ve:

            print(
                f" -> Skipped {filename}: {ve}"
            )

            error_log.append(
                f"Unsupported type: {filename}"
            )

        except Exception as e:

            print(
                f" -> Error processing {filename}: {e}"
            )

            error_log.append(
                f"Failed {filename}: {e}"
            )

    # ========================================================
    # PHASE 2: SEMANTIC PROCESSING
    # ========================================================

    print("\n--- Phase 2: Semantic Indexing ---")

    if text_corpus:

        try:

            processor = SemanticProcessor(
                index_dir=INDEX_DIR
            )

            processor.chunk_and_embed(
                text_corpus
            )

        except Exception as e:

            print(
                f"Error during semantic processing and indexing: {e}"
            )

    else:

        print(
            "No readable text was extracted. "
            "Skipping indexing phase."
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n--- Ingestion Summary ---")

    print(
        f"Total files detected: {len(all_files)}"
    )

    print(
        f"Files successfully extracted: {len(text_corpus)}"
    )

    if error_log:

        print("\nErrors encountered:")

        for err in error_log:
            print(f"- {err}")

    else:

        print(
            "\nAll files processed successfully! "
            "You are ready to run the query pipeline."
        )


if __name__ == "__main__":
    main()