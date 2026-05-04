"""
Entity Normalization Module
Maps extracted entities to canonical/standard forms.
"""

# Disease normalization
DISEASE_NORMALIZATION = {
    # Diabetes variants
    "high blood sugar": "Diabetes Mellitus",
    "elevated blood sugar": "Diabetes Mellitus",
    "hyperglycemia": "Diabetes Mellitus",
    "diabetes": "Diabetes Mellitus",
    "diabetic": "Diabetes Mellitus",
    "type 2 diabetes": "Type 2 Diabetes Mellitus",
    "type ii diabetes": "Type 2 Diabetes Mellitus",
    "type 1 diabetes": "Type 1 Diabetes Mellitus",
    "type i diabetes": "Type 1 Diabetes Mellitus",
    "insulin dependent diabetes mellitus": "Type 1 Diabetes Mellitus",
    "non-insulin dependent diabetes mellitus": "Type 2 Diabetes Mellitus",
    # Hypertension
    "high blood pressure": "Hypertension",
    "elevated blood pressure": "Hypertension",
    "htn": "Hypertension",
    # Heart conditions
    "heart attack": "Myocardial Infarction",
    "mi": "Myocardial Infarction",
    "heart failure": "Congestive Heart Failure",
    "chf": "Congestive Heart Failure",
    "atrial fibrillation": "Atrial Fibrillation",
    "a-fib": "Atrial Fibrillation",
    "afib": "Atrial Fibrillation",
    "coronary artery disease": "Coronary Artery Disease",
    "cad": "Coronary Artery Disease",
    # Respiratory
    "chronic obstructive pulmonary disease": "COPD",
    "copd": "COPD",
    "asthma": "Asthma",
    "pneumonia": "Pneumonia",
    "upper respiratory infection": "Upper Respiratory Infection",
    "uri": "Upper Respiratory Infection",
    # Kidney
    "chronic kidney disease": "Chronic Kidney Disease",
    "ckd": "Chronic Kidney Disease",
    "renal failure": "Renal Failure",
    "kidney failure": "Renal Failure",
    # Neurological
    "stroke": "Cerebrovascular Accident",
    "cva": "Cerebrovascular Accident",
    "cerebrovascular accident": "Cerebrovascular Accident",
    "tia": "Transient Ischemic Attack",
    "transient ischemic attack": "Transient Ischemic Attack",
    "seizure": "Seizure Disorder",
    "epilepsy": "Epilepsy",
    # GI
    "gastroesophageal reflux disease": "GERD",
    "gerd": "GERD",
    "acid reflux": "GERD",
    "peptic ulcer": "Peptic Ulcer Disease",
    "peptic ulcer disease": "Peptic Ulcer Disease",
    # Musculoskeletal
    "rheumatoid arthritis": "Rheumatoid Arthritis",
    "osteoarthritis": "Osteoarthritis",
    "degenerative joint disease": "Osteoarthritis",
    # Infections
    "urinary tract infection": "Urinary Tract Infection",
    "uti": "Urinary Tract Infection",
    "cellulitis": "Cellulitis",
    # Cancer
    "cancer": "Malignancy",
    "carcinoma": "Carcinoma",
    "tumor": "Neoplasm",
    "malignancy": "Malignancy",
    # Other
    "hypothyroidism": "Hypothyroidism",
    "hyperthyroidism": "Hyperthyroidism",
    "anemia": "Anemia",
    "obesity": "Obesity",
    "depression": "Major Depressive Disorder",
    "anxiety": "Anxiety Disorder",
    "sleep apnea": "Obstructive Sleep Apnea",
    "obstructive sleep apnea": "Obstructive Sleep Apnea",
    "osa": "Obstructive Sleep Apnea",
    "hyperlipidemia": "Hyperlipidemia",
    "high cholesterol": "Hyperlipidemia",
    "elevated cholesterol": "Hyperlipidemia",
    "gout": "Gout",
    "fibromyalgia": "Fibromyalgia",
    "allergic rhinitis": "Allergic Rhinitis",
    "allergies": "Allergic Rhinitis",
}

# Drug normalization
DRUG_NORMALIZATION = {
    # Analgesics
    "paracetamol": "Acetaminophen",
    "tylenol": "Acetaminophen",
    "acetaminophen": "Acetaminophen",
    "apap": "Acetaminophen",
    "ibuprofen": "Ibuprofen",
    "advil": "Ibuprofen",
    "motrin": "Ibuprofen",
    "naproxen": "Naproxen",
    "aleve": "Naproxen",
    "aspirin": "Aspirin",
    "asa": "Aspirin",
    # Antibiotics
    "amoxicillin": "Amoxicillin",
    "augmentin": "Amoxicillin-Clavulanate",
    "amoxicillin-clavulanate": "Amoxicillin-Clavulanate",
    "azithromycin": "Azithromycin",
    "zithromax": "Azithromycin",
    "z-pack": "Azithromycin",
    "ciprofloxacin": "Ciprofloxacin",
    "cipro": "Ciprofloxacin",
    "doxycycline": "Doxycycline",
    "penicillin": "Penicillin",
    "pcn": "Penicillin",
    "trimethoprim-sulfamethoxazole": "Trimethoprim-Sulfamethoxazole",
    "bactrim": "Trimethoprim-Sulfamethoxazole",
    "tmp-smx": "Trimethoprim-Sulfamethoxazole",
    "unasyn": "Ampicillin-Sulbactam",
    # Cardiovascular
    "lisinopril": "Lisinopril",
    "enalapril": "Enalapril",
    "metoprolol": "Metoprolol",
    "lopressor": "Metoprolol",
    "atenolol": "Atenolol",
    "amlodipine": "Amlodipine",
    "norvasc": "Amlodipine",
    "hydrochlorothiazide": "Hydrochlorothiazide",
    "hctz": "Hydrochlorothiazide",
    "furosemide": "Furosemide",
    "lasix": "Furosemide",
    "warfarin": "Warfarin",
    "coumadin": "Warfarin",
    "digoxin": "Digoxin",
    "nitroglycerin": "Nitroglycerin",
    "ntg": "Nitroglycerin",
    "diovan": "Valsartan",
    "valsartan": "Valsartan",
    "zaroxolyn": "Metolazone",
    "metolazone": "Metolazone",
    "pacerone": "Amiodarone",
    "amiodarone": "Amiodarone",
    # Statins
    "atorvastatin": "Atorvastatin",
    "lipitor": "Atorvastatin",
    "simvastatin": "Simvastatin",
    "zocor": "Simvastatin",
    "rosuvastatin": "Rosuvastatin",
    "crestor": "Rosuvastatin",
    "tricor": "Fenofibrate",
    "fenofibrate": "Fenofibrate",
    # Diabetes
    "metformin": "Metformin",
    "glucophage": "Metformin",
    "insulin": "Insulin",
    "humulin": "Insulin",
    "glipizide": "Glipizide",
    "glucotrol": "Glipizide",
    "glimepiride": "Glimepiride",
    "amaryl": "Glimepiride",
    # Respiratory
    "albuterol": "Albuterol",
    "proventil": "Albuterol",
    "ventolin": "Albuterol",
    "salmeterol": "Salmeterol",
    "fluticasone": "Fluticasone",
    "advair": "Fluticasone-Salmeterol",
    "montelukast": "Montelukast",
    "singulair": "Montelukast",
    "nasonex": "Mometasone",
    "mometasone": "Mometasone",
    # GI
    "omeprazole": "Omeprazole",
    "prilosec": "Omeprazole",
    "pantoprazole": "Pantoprazole",
    "protonix": "Pantoprazole",
    "prevacid": "Lansoprazole",
    "lansoprazole": "Lansoprazole",
    "ranitidine": "Ranitidine",
    "zantac": "Ranitidine",
    # Antihistamines
    "loratadine": "Loratadine",
    "claritin": "Loratadine",
    "cetirizine": "Cetirizine",
    "zyrtec": "Cetirizine",
    "fexofenadine": "Fexofenadine",
    "allegra": "Fexofenadine",
    # Neurological / Psych
    "gabapentin": "Gabapentin",
    "neurontin": "Gabapentin",
    "pregabalin": "Pregabalin",
    "lyrica": "Pregabalin",
    "sertraline": "Sertraline",
    "zoloft": "Sertraline",
    "fluoxetine": "Fluoxetine",
    "prozac": "Fluoxetine",
    "amitriptyline": "Amitriptyline",
    "hydrocodone": "Hydrocodone",
    "tussionex": "Hydrocodone",
    "oxycodone": "Oxycodone",
    "morphine": "Morphine",
    "diprivan": "Propofol",
    "propofol": "Propofol",
    "solu-medrol": "Methylprednisolone",
    "methylprednisolone": "Methylprednisolone",
    "prednisone": "Prednisone",
    "chantix": "Varenicline",
    "varenicline": "Varenicline",
    # Contraceptives
    "ortho tri-cyclen": "Norgestimate-Ethinyl Estradiol",
}

# Procedure normalization
PROCEDURE_NORMALIZATION = {
    "ekg": "Electrocardiogram",
    "ecg": "Electrocardiogram",
    "electrocardiogram": "Electrocardiogram",
    "cbc": "Complete Blood Count",
    "complete blood count": "Complete Blood Count",
    "bmp": "Basic Metabolic Panel",
    "basic metabolic panel": "Basic Metabolic Panel",
    "cmp": "Comprehensive Metabolic Panel",
    "comprehensive metabolic panel": "Comprehensive Metabolic Panel",
    "chest x-ray": "Chest X-Ray",
    "cxr": "Chest X-Ray",
    "ct scan": "CT Scan",
    "computed tomography": "CT Scan",
    "mri": "MRI",
    "magnetic resonance imaging": "MRI",
    "ultrasound": "Ultrasound",
    "echocardiogram": "Echocardiogram",
    "echo": "Echocardiogram",
    "colonoscopy": "Colonoscopy",
    "endoscopy": "Endoscopy",
    "upper endoscopy": "Upper Endoscopy",
    "biopsy": "Biopsy",
    "lumbar puncture": "Lumbar Puncture",
    "lp": "Lumbar Puncture",
    "urinalysis": "Urinalysis",
    "ua": "Urinalysis",
    "blood culture": "Blood Culture",
    "thyroid function test": "Thyroid Function Test",
    "liver function test": "Liver Function Test",
    "pulmonary function test": "Pulmonary Function Test",
    "arterial blood gas": "Arterial Blood Gas",
    "abg": "Arterial Blood Gas",
    "sleep study": "Polysomnography",
    "polysomnography": "Polysomnography",
    "stress test": "Cardiac Stress Test",
    "cardiac catheterization": "Cardiac Catheterization",
    "coronary angiography": "Coronary Angiography",
    "cabg": "Coronary Artery Bypass Graft",
    "coronary artery bypass graft": "Coronary Artery Bypass Graft",
    "cholecystectomy": "Cholecystectomy",
    "appendectomy": "Appendectomy",
    "hysterectomy": "Hysterectomy",
    "tracheostomy": "Tracheostomy",
    "intubation": "Endotracheal Intubation",
    "intubated": "Endotracheal Intubation",
    "cardioversion": "Cardioversion",
    "dialysis": "Hemodialysis",
    "hemodialysis": "Hemodialysis",
}

# Anatomy normalization
ANATOMY_NORMALIZATION = {
    "heart": "Heart",
    "cardiac": "Heart",
    "lung": "Lung",
    "lungs": "Lungs",
    "pulmonary": "Lung",
    "liver": "Liver",
    "hepatic": "Liver",
    "kidney": "Kidney",
    "kidneys": "Kidneys",
    "renal": "Kidney",
    "brain": "Brain",
    "cerebral": "Brain",
    "stomach": "Stomach",
    "gastric": "Stomach",
    "intestine": "Intestine",
    "bowel": "Intestine",
    "colon": "Colon",
    "spine": "Spine",
    "spinal": "Spine",
    "knee": "Knee",
    "hip": "Hip",
    "shoulder": "Shoulder",
    "chest": "Chest",
    "abdomen": "Abdomen",
    "abdominal": "Abdomen",
    "throat": "Throat",
    "neck": "Neck",
    "head": "Head",
    "eye": "Eye",
    "eyes": "Eyes",
    "ear": "Ear",
    "ears": "Ears",
    "nose": "Nose",
    "nasal": "Nose",
    "skin": "Skin",
    "bone": "Bone",
    "muscle": "Muscle",
    "joint": "Joint",
    "artery": "Artery",
    "vein": "Vein",
    "blood vessel": "Blood Vessel",
    "lymph node": "Lymph Node",
    "thyroid": "Thyroid Gland",
    "pancreas": "Pancreas",
    "gallbladder": "Gallbladder",
    "bladder": "Bladder",
    "prostate": "Prostate",
    "uterus": "Uterus",
    "ovary": "Ovary",
    "ovaries": "Ovaries",
    "appendix": "Appendix",
    "tonsil": "Tonsil",
    "tonsils": "Tonsils",
    "airway": "Airway",
    "trachea": "Trachea",
    "esophagus": "Esophagus",
    "aorta": "Aorta",
    "ventricle": "Ventricle",
    "atrium": "Atrium",
    "mitral valve": "Mitral Valve",
    "aortic valve": "Aortic Valve",
    "tricuspid valve": "Tricuspid Valve",
    "pulmonary valve": "Pulmonary Valve",
}

# Combined normalization map
ALL_NORMALIZATION = {}
ALL_NORMALIZATION.update(DISEASE_NORMALIZATION)
ALL_NORMALIZATION.update(DRUG_NORMALIZATION)
ALL_NORMALIZATION.update(PROCEDURE_NORMALIZATION)
ALL_NORMALIZATION.update(ANATOMY_NORMALIZATION)


def normalize_entity(entity_text: str, entity_type: str = None) -> str:
    """
    Normalize an extracted entity to its canonical form.
    
    Args:
        entity_text: The raw entity text
        entity_type: Optional entity type to narrow normalization scope
    
    Returns:
        Normalized entity string
    """
    if not entity_text:
        return entity_text

    key = entity_text.lower().strip()

    # Try type-specific normalization first
    if entity_type:
        type_map = {
            "DISEASE": DISEASE_NORMALIZATION,
            "DRUG": DRUG_NORMALIZATION,
            "PROCEDURE": PROCEDURE_NORMALIZATION,
            "ANATOMY": ANATOMY_NORMALIZATION,
        }
        specific_map = type_map.get(entity_type.upper(), {})
        if key in specific_map:
            return specific_map[key]

    # Fall back to combined map
    if key in ALL_NORMALIZATION:
        return ALL_NORMALIZATION[key]

    # Return title-cased original if no match
    return entity_text.strip()


def normalize_entities_list(entities: list) -> list:
    """
    Normalize a list of (entity_text, entity_type) tuples.
    Returns list of (entity_text, entity_type, normalized_form) tuples.
    """
    result = []
    for item in entities:
        if len(item) == 2:
            text, etype = item
            normalized = normalize_entity(text, etype)
            result.append((text, etype, normalized))
        elif len(item) == 3:
            text, etype, _ = item
            normalized = normalize_entity(text, etype)
            result.append((text, etype, normalized))
    return result
