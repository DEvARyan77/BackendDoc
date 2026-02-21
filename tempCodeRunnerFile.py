import pandas as pd
import ast
import json

print("Loading datasets...")

# 1. Load detailed diseases file (the ~‑separated one)
#    Important: use '~' as separator, and handle possible quoting issues.
df_detailed = pd.read_csv('clean.csv', sep='~', dtype=str, engine='python')

# Convert stringified lists to actual Python lists
list_cols = ['symptoms', 'commonTestsAndProcedures', 'commonMedications']
for col in list_cols:
    if col in df_detailed.columns:
        df_detailed[col] = df_detailed[col].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and x.startswith('[') else []
        )

# 2. Load Disease_Description.csv
df_desc = pd.read_csv('Disease_Description.csv')

# 3. Load medicines data
df_meds = pd.read_csv('Medicine_Details.csv')

print("Merging data...")

disease_info = {}

# Start with df_detailed
for _, row in df_detailed.iterrows():
    name = row['name'].strip()
    disease_info[name] = {
        'description': row.get('Desc', ''),
        'symptoms_list': row.get('symptoms', []),
        'tests': row.get('commonTestsAndProcedures', []),
        'medications_list': row.get('commonMedications', []),
        'riskGroups': row.get('whoIsAtRiskDesc', ''),
        'symptomsDesc': row.get('symptomsDesc', '')
    }

# Add/merge from Disease_Description.csv
for _, row in df_desc.iterrows():
    name = row['Disease'].strip()
    if name in disease_info:
        disease_info[name]['description'] = disease_info[name]['description'] or row.get('Description', '')
        disease_info[name]['symptoms_text'] = row.get('Symptoms', '')
        disease_info[name]['treatment'] = row.get('Treatment', '')
    else:
        disease_info[name] = {
            'description': row.get('Description', ''),
            'symptoms_text': row.get('Symptoms', ''),
            'treatment': row.get('Treatment', ''),
            'symptoms_list': [],
            'tests': [],
            'medications_list': [],
            'riskGroups': '',
            'symptomsDesc': ''
        }

# Link medications from the medicines file (simple keyword matching)
for _, row in df_meds.iterrows():
    medicine_name = row.get('Medicine Name', '')
    uses = row.get('Uses', '')
    if pd.isna(uses):
        continue
    uses_lower = uses.lower()
    for disease in list(disease_info.keys()):
        if disease.lower() in uses_lower:
            if 'medications' not in disease_info[disease]:
                disease_info[disease]['medications'] = []
            disease_info[disease]['medications'].append(medicine_name)

# Merge medication lists (from both sources)
for disease in disease_info:
    meds = set(disease_info[disease].get('medications', []))
    meds.update(disease_info[disease].get('medications_list', []))
    disease_info[disease]['medications'] = list(meds)
    # Clean up temporary keys
    disease_info[disease].pop('medications_list', None)

    # Combine symptoms lists and text into one string
    symptoms_parts = []
    if disease_info[disease].get('symptoms_list'):
        symptoms_parts.extend(disease_info[disease]['symptoms_list'])
    if disease_info[disease].get('symptoms_text'):
        symptoms_parts.append(disease_info[disease]['symptoms_text'])
    disease_info[disease]['symptoms'] = ', '.join(symptoms_parts) if symptoms_parts else ''
    disease_info[disease].pop('symptoms_list', None)
    disease_info[disease].pop('symptoms_text', None)

# Save to JSON
with open('disease_db.json', 'w', encoding='utf-8') as f:
    json.dump(disease_info, f, indent=2, ensure_ascii=False)

print("disease_db.json created successfully.")