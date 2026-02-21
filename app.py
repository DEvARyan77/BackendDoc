import pandas as pd
import ast
import json

print("Loading datasets...")

# 1. Load Symptom2Disease to get all possible disease labels
df_s2d = pd.read_csv('clean.csv', index_col=0, encoding='utf-8')
all_diseases = set(df_s2d['label'].unique())
print(f"Total diseases in training set: {len(all_diseases)}")

# 2. Load detailed diseases file (~ separated)
df_detailed = pd.read_csv('clean1.csv', sep='~', dtype=str, engine='python')
list_cols = ['symptoms', 'commonTestsAndProcedures', 'commonMedications']
for col in list_cols:
    if col in df_detailed.columns:
        df_detailed[col] = df_detailed[col].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and x.startswith('[') else []
        )

# 3. Load Disease_Description.csv
df_desc = pd.read_csv('Disease_Description.csv')

# 4. Load medicines data
df_meds = pd.read_csv('Medicine_Details.csv')

print("Building disease database...")

# Initialize disease_info for every disease in the training set
disease_info = {disease: {
    'description': '',
    'symptoms': '',
    'treatment': '',
    'tests': [],
    'medications': [],
    'riskGroups': ''
} for disease in all_diseases}

# Merge data from df_detailed
for _, row in df_detailed.iterrows():
    name = row['name'].strip()
    if name in disease_info:
        disease_info[name]['description'] = row.get('Desc', '')
        disease_info[name]['tests'] = row.get('commonTestsAndProcedures', [])
        disease_info[name]['medications'] = row.get('commonMedications', [])
        disease_info[name]['riskGroups'] = row.get('whoIsAtRiskDesc', '')
        # Also add symptoms list (will combine later)
        disease_info[name]['symptoms_list'] = row.get('symptoms', [])
    else:
        # If the disease isn't in the training set, we still add it (optional)
        disease_info[name] = {
            'description': row.get('Desc', ''),
            'symptoms': '',
            'treatment': '',
            'tests': row.get('commonTestsAndProcedures', []),
            'medications': row.get('commonMedications', []),
            'riskGroups': row.get('whoIsAtRiskDesc', ''),
            'symptoms_list': row.get('symptoms', [])
        }

# Merge data from df_desc
for _, row in df_desc.iterrows():
    name = row['Disease'].strip()
    if name in disease_info:
        disease_info[name]['description'] = disease_info[name]['description'] or row.get('Description', '')
        disease_info[name]['symptoms_text'] = row.get('Symptoms', '')
        disease_info[name]['treatment'] = row.get('Treatment', '')
    else:
        disease_info[name] = {
            'description': row.get('Description', ''),
            'symptoms': row.get('Symptoms', ''),
            'treatment': row.get('Treatment', ''),
            'tests': [],
            'medications': [],
            'riskGroups': ''
        }

# Link medications from medicines file using disease name in "Uses" column
print("Linking medications from medicines file...")
for _, row in df_meds.iterrows():
    medicine_name = row.get('Medicine Name', '')
    uses = row.get('Uses', '')
    if pd.isna(uses) or not medicine_name:
        continue
    uses_lower = uses.lower()
    # Check each disease name
    for disease in disease_info.keys():
        if disease.lower() in uses_lower:
            disease_info[disease]['medications'].append(medicine_name)

# Combine medication lists (remove duplicates) and limit to 50
for disease in disease_info:
    meds = set(disease_info[disease].get('medications', []))
    # Also include medications from detailed file if any
    detailed_meds = disease_info[disease].get('medications_list', [])
    if detailed_meds:
        meds.update(detailed_meds)
    # Convert back to list, sort, and take first 50
    meds = sorted(list(meds))[:50]
    disease_info[disease]['medications'] = meds

    # Combine symptoms: if we have a list, join it; if we have text, add it
    symptoms_parts = []
    if 'symptoms_list' in disease_info[disease]:
        symptoms_parts.extend(disease_info[disease]['symptoms_list'])
    if 'symptoms_text' in disease_info[disease]:
        symptoms_parts.append(disease_info[disease]['symptoms_text'])
    disease_info[disease]['symptoms'] = ', '.join(symptoms_parts) if symptoms_parts else ''
    
    # Clean up temporary keys
    for key in ['symptoms_list', 'symptoms_text', 'medications_list']:
        disease_info[disease].pop(key, None)

# Save to JSON
with open('disease_db.json', 'w', encoding='utf-8') as f:
    json.dump(disease_info, f, indent=2, ensure_ascii=False)

print(f"disease_db.json created with {len(disease_info)} diseases.")
print("Sample keys:", list(disease_info.keys())[:5])