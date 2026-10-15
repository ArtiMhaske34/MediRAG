import os

os.makedirs("source_documents", exist_ok=True)

# 1. Mock Electronic Health Record (EHR)
ehr_content = """
PATIENT RECORD: MRN-88492
Name: Rajesh Sharma
Age: 45 | Gender: Male
Date of Visit: 2025-10-14
Chief Complaint: Patient reports persistent chest pain radiating to the left arm, accompanied by shortness of breath and mild diaphoresis for the past 24 hours.
Medical History: Type 2 Diabetes Mellitus (diagnosed 2020), Hypertension.
Current Medications: Metformin 500mg BID, Amlodipine 5mg QD.
Assessment: Rule out Acute Coronary Syndrome (ACS).
Plan: Advised immediate ECG and Troponin levels. Prescribed sublingual Nitroglycerin PRN for chest pain.
"""
with open("source_documents/EHR_Rajesh_Sharma.txt", "w") as f:
    f.write(ehr_content)

# 2. Mock Lab Report
lab_content = """
LABORATORY RESULTS
Patient: Swapnali Kulkarni | MRN: 55210
Test Date: 2025-11-02
TEST                RESULT      REFERENCE RANGE
Hemoglobin (Hb)     11.2 g/dL   (12.0 - 15.5 g/dL) *LOW
WBC Count           8.5 K/uL    (4.5 - 11.0 K/uL)
Platelets           210 K/uL    (150 - 450 K/uL)
Serum Creatinine    0.9 mg/dL   (0.6 - 1.1 mg/dL)
Notes: Patient presents with mild anemia. Recommend iron supplementation and follow-up in 3 months.
"""
with open("source_documents/Lab_Results_Swapnali.txt", "w") as f:
    f.write(lab_content)

print("Mock medical data generated in source_documents/!")