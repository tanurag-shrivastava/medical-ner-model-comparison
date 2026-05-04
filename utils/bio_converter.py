"""
BIO Tagging Converter
Converts tokenized sentences to BIO-tagged format using medical dictionaries.
"""

import re
from utils.preprocessing import LABEL_LIST, LABEL2ID, ID2LABEL

# ─── Medical Gazetteers ───────────────────────────────────────────────────────

DISEASE_TERMS = [
    # Metabolic / Endocrine
    "diabetes mellitus", "diabetes", "diabetic",
    "type 1 diabetes", "type 2 diabetes",
    "hypothyroidism", "hyperthyroidism", "thyroid disease",
    "hyperlipidemia", "hypercholesterolemia", "obesity",
    "high blood pressure", "high blood sugar", "high cholesterol",
    "elevated blood pressure", "elevated blood sugar",
    # Cardiovascular
    "hypertension", "hypertensive",
    "coronary artery disease", "heart failure", "congestive heart failure",
    "myocardial infarction", "heart attack",
    "atrial fibrillation", "arrhythmia",
    "pulmonary embolism", "deep vein thrombosis",
    "peripheral vascular disease", "venous insufficiency",
    "thrombophlebitis", "phlebitis",
    # Respiratory (diagnosed conditions, NOT symptoms)
    "chronic obstructive pulmonary disease", "copd", "asthma", "pneumonia",
    "bronchitis", "sinusitis", "allergic rhinitis",
    "viral syndrome", "viral respiratory illness", "viral infection",
    "bacterial infection", "upper respiratory infection",
    # Renal
    "chronic kidney disease", "renal failure", "kidney failure",
    # Neurological
    "cerebrovascular accident", "stroke", "transient ischemic attack",
    "multiple sclerosis", "parkinson disease", "alzheimer disease",
    "epilepsy", "seizure disorder",
    "pseudotumor cerebri", "meningitis", "encephalitis",
    # GI (diagnosed conditions)
    "gastroesophageal reflux disease", "gerd",
    "peptic ulcer disease", "peptic ulcer",
    "inflammatory bowel disease", "crohn disease", "ulcerative colitis",
    "cirrhosis", "hepatitis", "fatty liver", "pancreatitis",
    "gallstones", "cholelithiasis", "cholecystitis",
    "hiatal hernia", "hernia", "hemorrhoids",
    # Musculoskeletal
    "rheumatoid arthritis", "osteoarthritis", "degenerative joint disease",
    "osteoporosis", "gout", "fibromyalgia",
    # Infections
    "urinary tract infection", "cellulitis", "sepsis", "bacteremia",
    # Oncology
    "cancer", "carcinoma", "malignancy", "tumor", "lymphoma", "leukemia",
    # Psychiatric (diagnosed conditions)
    "depression", "anxiety", "bipolar disorder", "schizophrenia",
    # Sleep (diagnosed conditions)
    "obstructive sleep apnea", "sleep apnea", "sleep disorder",
    "restless leg syndrome",
    # Blood
    "anemia",
    # Misc diagnosed conditions
    "migraine",
    "infection",
]

SYMPTOM_TERMS = [
    # Respiratory symptoms
    "shortness of breath", "difficulty breathing", "dyspnea", "breathlessness",
    "wheezing", "stridor", "cough", "productive cough", "dry cough",
    "hemoptysis", "coughing blood",
    # Chest symptoms
    "chest pain", "chest tightness", "chest pressure", "chest discomfort",
    # Pain
    "pain", "ache", "aching", "soreness", "tenderness",
    "abdominal pain", "stomach pain", "cramping",
    "back pain", "lower back pain", "neck pain",
    "joint pain", "joint stiffness", "muscle pain",
    "knee pain", "hip pain", "shoulder pain",
    "headache", "head pain",
    "ear pain", "throat pain",
    # GI symptoms
    "nausea", "vomiting", "nausea and vomiting",
    "diarrhea", "constipation", "bloating",
    "rectal bleeding", "blood in stool",
    "hematemesis", "vomiting blood",
    "melena", "black stool",
    "jaundice", "yellow skin",
    "ascites",
    # Neurological symptoms
    "dizziness", "vertigo", "lightheadedness",
    "confusion", "disorientation", "altered mental status", "memory loss",
    "numbness", "tingling", "paresthesia",
    "syncope", "fainting", "presyncope",
    "weakness", "fatigue", "tiredness", "malaise",
    "sluggish",
    # Visual / Auditory
    "blurred vision", "double vision", "vision loss",
    "hearing loss", "tinnitus",
    # Swallowing
    "difficulty swallowing", "dysphagia",
    # Constitutional
    "fever", "chills", "night sweats",
    "weight loss", "weight gain", "appetite loss",
    "diaphoresis", "sweating",
    # Skin
    "rash", "itching", "pruritus", "erythema", "erythematous",
    "pallor", "cyanosis",
    # Cardiovascular symptoms
    "palpitations", "racing heart",
    "tachycardia", "bradycardia", "hypotension", "tachypnea",
    # ENT symptoms
    "sore throat", "hoarseness",
    "runny nose", "nasal congestion", "nasal discharge",
    "snoring",
    # Urinary symptoms
    "urinary frequency", "urgency", "dysuria",
    "hematuria", "blood in urine", "proteinuria",
    "urinary incontinence", "incontinence",
    # Swelling
    "swelling", "edema", "pitting edema", "puffiness",
    "lymphadenopathy", "adenopathy",
    # Psychiatric symptoms
    "anxiety", "depression", "mood changes",
    "irritability", "agitation", "insomnia", "difficulty sleeping",
    # Metabolic symptoms
    "polyuria", "polydipsia", "polyphagia",
    # Misc
    "drainage", "discharge",
]

DRUG_TERMS = [
    "aspirin", "ibuprofen", "naproxen", "acetaminophen", "paracetamol", "tylenol",
    "morphine", "oxycodone", "hydrocodone", "codeine", "tramadol", "fentanyl",
    "amoxicillin", "augmentin", "azithromycin", "ciprofloxacin", "doxycycline",
    "penicillin", "cephalexin", "clindamycin", "metronidazole", "vancomycin",
    "lisinopril", "enalapril", "ramipril", "captopril",
    "metoprolol", "atenolol", "carvedilol", "bisoprolol",
    "amlodipine", "nifedipine", "diltiazem", "verapamil",
    "hydrochlorothiazide", "furosemide", "spironolactone", "metolazone",
    "warfarin", "heparin", "enoxaparin", "rivaroxaban", "apixaban",
    "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
    "metformin", "glipizide", "glimepiride", "sitagliptin",
    "insulin", "lantus", "humalog", "novolog", "humulin",
    "albuterol", "salmeterol", "tiotropium", "ipratropium",
    "fluticasone", "budesonide", "prednisone", "methylprednisolone",
    "omeprazole", "pantoprazole", "lansoprazole", "esomeprazole", "ranitidine",
    "loratadine", "cetirizine", "fexofenadine", "diphenhydramine",
    "sertraline", "fluoxetine", "paroxetine", "escitalopram", "citalopram",
    "amitriptyline", "nortriptyline", "duloxetine", "venlafaxine",
    "gabapentin", "pregabalin", "phenytoin", "carbamazepine", "valproate",
    "levothyroxine", "synthroid",
    "digoxin", "nitroglycerin", "isosorbide",
    "clopidogrel", "plavix", "ticagrelor",
    "montelukast", "singulair",
    "methotrexate", "hydroxychloroquine", "sulfasalazine",
    "allopurinol", "colchicine",
    "sildenafil", "tadalafil",
    "tamsulosin", "finasteride",
    "alendronate", "risedronate",
    "calcium", "vitamin d", "iron", "folic acid",
    "potassium", "magnesium",
    "diovan", "valsartan", "losartan", "cozaar",
    "zaroxolyn", "lasix", "pacerone", "amiodarone",
    "prevacid", "proventil", "unasyn", "solu-medrol",
    "neurontin", "crestor", "tricor", "coumadin",
    "allegra", "zyrtec", "claritin", "nasonex",
    "chantix", "varenicline",
    "diprivan", "propofol",
    "tussionex",
    "ortho tri-cyclen",
    "antibiotic", "antibiotics",
    "steroid", "corticosteroid",
    "inhaler", "nebulizer",
]

PROCEDURE_TERMS = [
    "electrocardiogram", "ekg", "ecg",
    "complete blood count", "cbc",
    "basic metabolic panel", "bmp",
    "comprehensive metabolic panel", "cmp",
    "chest x-ray", "x-ray", "xray",
    "ct scan", "computed tomography",
    "mri", "magnetic resonance imaging",
    "ultrasound", "sonogram",
    "echocardiogram", "echo",
    "colonoscopy", "endoscopy", "upper endoscopy",
    "biopsy", "fine needle aspiration",
    "lumbar puncture", "spinal tap",
    "urinalysis", "urine culture",
    "blood culture", "culture",
    "thyroid function test", "thyroid function tests",
    "liver function test", "liver function tests",
    "pulmonary function test", "pulmonary function tests",
    "arterial blood gas",
    "sleep study", "polysomnography",
    "stress test", "cardiac stress test",
    "cardiac catheterization", "catheterization",
    "coronary angiography", "angiography",
    "coronary artery bypass graft", "cabg",
    "percutaneous coronary intervention", "pci",
    "cholecystectomy", "appendectomy", "hysterectomy",
    "tracheostomy", "intubation", "extubation",
    "cardioversion", "defibrillation",
    "dialysis", "hemodialysis",
    "surgery", "operation", "procedure",
    "biopsy", "excision", "resection",
    "transfusion", "blood transfusion",
    "vaccination", "immunization",
    "physical examination", "examination",
    "laboratory", "lab work", "lab tests",
    "imaging", "scan", "test", "testing",
    "workup", "evaluation",
    "h. pylori testing", "glycosylated hemoglobin",
    "fasting blood sugar",
    "reconstructive surgery",
    "orthopedic surgery", "knee surgery",
]

ANATOMY_TERMS = [
    "heart", "cardiac", "myocardium", "pericardium",
    "lung", "lungs", "pulmonary", "bronchus", "bronchi", "trachea", "airway",
    "liver", "hepatic", "gallbladder", "bile duct",
    "kidney", "kidneys", "renal", "ureter", "bladder",
    "brain", "cerebral", "cerebrum", "cerebellum", "brainstem",
    "spinal cord", "spine", "vertebra", "vertebrae",
    "stomach", "gastric", "esophagus", "duodenum", "jejunum", "ileum",
    "colon", "rectum", "anus", "intestine", "bowel",
    "pancreas", "pancreatic", "spleen", "splenic",
    "thyroid", "parathyroid", "adrenal", "pituitary",
    "aorta", "aortic", "coronary artery", "carotid artery",
    "left ventricle", "right ventricle", "left atrium", "right atrium",
    "mitral valve", "aortic valve", "tricuspid valve", "pulmonary valve",
    "knee", "hip", "shoulder", "elbow", "wrist", "ankle", "foot",
    "femur", "tibia", "fibula", "humerus", "radius", "ulna",
    "chest", "thorax", "thoracic",
    "abdomen", "abdominal", "pelvis", "pelvic",
    "head", "skull", "cranium",
    "neck", "cervical",
    "arm", "leg", "extremity", "extremities",
    "hand", "finger", "toe",
    "eye", "eyes", "retina", "cornea", "lens",
    "ear", "ears", "cochlea",
    "nose", "nasal", "sinus", "sinuses",
    "throat", "pharynx", "larynx",
    "mouth", "oral", "tongue", "teeth",
    "skin", "dermis", "epidermis",
    "muscle", "tendon", "ligament", "cartilage",
    "bone", "marrow",
    "lymph node", "lymph nodes", "lymphatic",
    "blood vessel", "artery", "vein", "capillary",
    "nerve", "nerves", "neural",
    "prostate", "uterus", "ovary", "ovaries", "cervix",
    "breast", "mammary",
    "appendix",
    "tonsil", "tonsils", "adenoid",
    "diaphragm",
    "peritoneum", "pleura",
    "cranial nerves",
    "right upper quadrant", "left upper quadrant",
    "right lower quadrant", "left lower quadrant",
]


def build_gazetteer_lookup():
    """Build sorted gazetteer for multi-word matching."""
    gazetteers = {
        "DISEASE": sorted(set(DISEASE_TERMS), key=len, reverse=True),
        "SYMPTOM": sorted(set(SYMPTOM_TERMS), key=len, reverse=True),
        "DRUG": sorted(set(DRUG_TERMS), key=len, reverse=True),
        "PROCEDURE": sorted(set(PROCEDURE_TERMS), key=len, reverse=True),
        "ANATOMY": sorted(set(ANATOMY_TERMS), key=len, reverse=True),
    }
    return gazetteers


GAZETTEERS = build_gazetteer_lookup()


def tokens_to_bio(tokens: list, gazetteers: dict = None) -> list:
    """
    Convert a list of tokens to BIO tags using gazetteer matching.
    Returns list of (token, tag) pairs.
    """
    if gazetteers is None:
        gazetteers = GAZETTEERS

    n = len(tokens)
    tags = ["O"] * n
    lower_tokens = [t.lower() for t in tokens]

    # Priority: SYMPTOM before DISEASE so subjective complaints are never
    # overridden by the disease gazetteer. DRUG/PROCEDURE/ANATOMY are
    # unambiguous and can run first for efficiency.
    type_priority = ["DRUG", "PROCEDURE", "SYMPTOM", "ANATOMY", "DISEASE"]

    for entity_type in type_priority:
        terms = gazetteers[entity_type]
        for term in terms:
            term_tokens = term.lower().split()
            term_len = len(term_tokens)
            for i in range(n - term_len + 1):
                # Check if already tagged
                if any(tags[i + j] != "O" for j in range(term_len)):
                    continue
                # Check match
                if lower_tokens[i:i + term_len] == term_tokens:
                    tags[i] = f"B-{entity_type}"
                    for j in range(1, term_len):
                        tags[i + j] = f"I-{entity_type}"

    return list(zip(tokens, tags))


def bio_to_entities(bio_tagged: list) -> list:
    """
    Convert BIO-tagged list to entity spans.
    Returns list of (entity_text, entity_type) tuples.
    """
    entities = []
    current_entity = []
    current_type = None

    for token, tag in bio_tagged:
        if tag.startswith("B-"):
            if current_entity:
                entities.append((" ".join(current_entity), current_type))
            current_entity = [token]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append((" ".join(current_entity), current_type))
            current_entity = []
            current_type = None

    if current_entity:
        entities.append((" ".join(current_entity), current_type))

    return entities


def tag_sentences(sentences: list) -> list:
    """
    Tag a list of token lists with BIO labels.
    Returns list of (tokens, tags) pairs.
    """
    tagged = []
    for tokens in sentences:
        bio = tokens_to_bio(tokens)
        token_list = [t for t, _ in bio]
        tag_list = [t for _, t in bio]
        tagged.append((token_list, tag_list))
    return tagged
