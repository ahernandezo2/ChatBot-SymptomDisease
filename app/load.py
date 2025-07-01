import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data/Disease_symptom_and_patient_profile_dataset.csv")

# Clean symptom columns into list
symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
def extract_symptoms(row):
    return [symptom for symptom in symptom_cols if row[symptom] == "Yes"]
df["Symptoms_List"] = df.apply(extract_symptoms, axis=1)

# Add dummy specialist mapping if not already there
specialist_map = {
    # General Medicine / Internal Medicine
    "Common Cold": "General Practitioner",
    "Influenza": "General Practitioner",
    "Sinusitis": "General Practitioner",
    "Tonsillitis": "General Practitioner",
    "Otitis Media (Ear Infection)": "General Practitioner",
    "Gastroenteritis": "General Practitioner",
    "Cholera": "General Practitioner",
    "Typhoid Fever": "General Practitioner",
    "Dengue Fever": "Infectious Disease Specialist",
    "Malaria": "Infectious Disease Specialist",
    "Zika Virus": "Infectious Disease Specialist",
    "Tuberculosis": "Pulmonologist",
    "Ebola Virus": "Infectious Disease Specialist",
    "Rabies": "Infectious Disease Specialist",
    "COVID-19": "Infectious Disease Specialist",

    # Mental Health
    "Depression": "Psychiatrist",
    "Anxiety Disorders": "Psychiatrist",
    "Bipolar Disorder": "Psychiatrist",
    "Schizophrenia": "Psychiatrist",
    "Obsessive-Compulsive Disorde...": "Psychiatrist",
    "Eating Disorders (Anorexia,...": "Psychiatrist",
    "Tourette Syndrome": "Neurologist",

    # Endocrine
    "Diabetes": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypothyroidism": "Endocrinologist",
    "Hyperglycemia": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Polycystic Ovary Syndrome (PCOS)": "Endocrinologist",

    # Cardiovascular
    "Hypertension": "Cardiologist",
    "Coronary Artery Disease": "Cardiologist",
    "Myocardial Infarction (Heart...": "Cardiologist",
    "Hypertensive Heart Disease": "Cardiologist",
    "Atherosclerosis": "Cardiologist",

    # Pulmonary
    "Asthma": "Pulmonologist",
    "Bronchitis": "Pulmonologist",
    "Chronic Obstructive Pulmonary Disease (COPD)": "Pulmonologist",
    "Pneumonia": "Pulmonologist",
    "Pneumocystis Pneumonia (PCP)": "Pulmonologist",
    "Pneumothorax": "Pulmonologist",

    # Oncology
    "Lung Cancer": "Oncologist",
    "Breast Cancer": "Oncologist",
    "Colorectal Cancer": "Oncologist",
    "Ovarian Cancer": "Oncologist",
    "Pancreatic Cancer": "Oncologist",
    "Prostate Cancer": "Oncologist",
    "Kidney Cancer": "Oncologist",
    "Liver Cancer": "Oncologist",
    "Esophageal Cancer": "Oncologist",
    "Testicular Cancer": "Oncologist",
    "Thyroid Cancer": "Oncologist",
    "Melanoma": "Oncologist",
    "Lymphoma": "Oncologist",
    "Bladder Cancer": "Oncologist",
    
    # Neurology
    "Alzheimer's Disease": "Neurologist",
    "Parkinson's Disease": "Neurologist",
    "Epilepsy": "Neurologist",
    "Multiple Sclerosis": "Neurologist",
    "Dementia": "Neurologist",
    "Stroke": "Neurologist",
    "Cerebral Palsy": "Neurologist",
    "Autism Spectrum Disorder (ASD)": "Neurologist",
    "Down Syndrome": "Neurologist",
    "Klinefelter Syndrome": "Neurologist",
    "Turner Syndrome": "Neurologist",
    "Williams Syndrome": "Neurologist",
    "Spina Bifida": "Neurologist",
    "Prader-Willi Syndrome": "Neurologist",
    "Marfan Syndrome": "Geneticist",

    # Dermatology
    "Acne": "Dermatologist",
    "Eczema": "Dermatologist",
    "Psoriasis": "Dermatologist",

    # Rheumatology
    "Rheumatoid Arthritis": "Rheumatologist",
    "Systemic Lupus Erythematosus...": "Rheumatologist",
    "Osteoarthritis": "Rheumatologist",
    "Gout": "Rheumatologist",
    "Fibromyalgia": "Rheumatologist",

    # Gastroenterology
    "Crohn's Disease": "Gastroenterologist",
    "Ulcerative Colitis": "Gastroenterologist",
    "Cirrhosis": "Gastroenterologist",
    "Liver Disease": "Gastroenterologist",
    "Pancreatitis": "Gastroenterologist",
    "Diverticulitis": "Gastroenterologist",
    "Hepatitis": "Gastroenterologist",
    "Hepatitis B": "Gastroenterologist",

    # Hematology
    "Anemia": "Hematologist",
    "Sickle Cell Anemia": "Hematologist",
    "Hemophilia": "Hematologist",

    # Urology / Nephrology
    "Kidney Disease": "Nephrologist",
    "Chronic Kidney Disease": "Nephrologist",
    "Urinary Tract Infection": "Urologist",
    "Urinary Tract Infection (UTI)": "Urologist",

    # OB/GYN
    "Endometriosis": "Gynecologist",

    # Pediatrics
    "Measles": "Pediatrician",
    "Mumps": "Pediatrician",
    "Rubella": "Pediatrician",
    "Chickenpox": "Pediatrician",
    "Polio": "Pediatrician",
    "Appendicitis": "Pediatrician",

    # Ophthalmology
    "Cataracts": "Ophthalmologist",
    "Glaucoma": "Ophthalmologist",
    "Conjunctivitis (Pink Eye)": "Ophthalmologist",

    # ENT / Otolaryngology
    "Allergic Rhinitis": "ENT Specialist",
    
    # Surgery / Ortho
    "Scoliosis": "Orthopedic Surgeon",
    "Osteomyelitis": "Orthopedic Surgeon",
    
}
df["Specialist"] = df["Disease"].map(specialist_map).fillna("General Practitioner")

# Build document string for embeddings
df["Document"] = df.apply(lambda row: f"{row['Gender']}, {row['Age']} years old, has {', '.join(row['Symptoms_List'])}", axis=1)

# Encode and index
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["Document"].tolist(), show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
