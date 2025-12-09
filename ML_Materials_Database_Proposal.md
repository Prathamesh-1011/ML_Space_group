# Machine Learning-Guided Materials Stability Database
## A Hierarchical Approach to Accelerate Ground State Prediction

---

**Project Type:** Computational Materials Science + Machine Learning  
**Duration:** 16 weeks (4 months)  
**Difficulty:** Advanced (PhD/Postdoc level)  
**Team Size:** 3-5 students

---

## 📋 EXECUTIVE SUMMARY

**The Challenge:** Predicting the ground state structure of a new material composition requires testing all 230 possible space groups with expensive DFT calculations — a process taking 1000s of computing hours per composition.

**Our Solution:** Use machine learning to pre-screen the most probable space groups and predict stability, reducing DFT calculations by 10-100×, then validate with high-accuracy DFT to build a unified, quality-controlled materials database.

**Impact:** 
- Enable rapid materials discovery for energy, electronics, and catalysis
- Create a unified database reconciling discrepancies across existing databases (Materials Project, OQMD, AFLOW, JARVIS)
- Provide uncertainty-quantified predictions for researchers

**Novelty:**
- First ML-guided hierarchical approach combining multiple databases
- Incorporates uncertainty quantification and database reconciliation
- Extends to phase transition predictions (temperature/pressure dependence)

---

## 🎯 PROJECT MOTIVATION

### Why This Matters

Every new technology — better batteries, efficient solar cells, quantum computers — depends on discovering new materials with specific properties. The traditional approach is slow:

1. **Experimentalist synthesizes** → weeks/months per material
2. **Computational prediction** → 1000s of DFT calculations per composition
3. **Most calculations wasted** on unstable structures

### Current Database Landscape

Existing materials databases contain millions of DFT calculations:

| Database | # Materials | Strength | Limitation |
|----------|-------------|----------|------------|
| Materials Project | 154,000+ | Well-maintained, API | PBE functional only |
| OQMD | 1,000,000+ | Largest, hull distances | Mixed quality |
| AFLOW | 3,700,000+ | High-throughput | Complex access |
| JARVIS | 75,000+ | Multiple properties | Smaller coverage |
| NOMAD | 10,000,000+ | Repository-style | Less curated |

**The Problem:** 
- Same material has different volumes/energies across databases (functional differences)
- No systematic way to predict phase transitions (T, P dependence)
- Computing convex hulls requires testing all possible structures

### The Gap We're Filling

**No existing approach systematically uses ML to:**
1. Pre-screen space groups before expensive DFT
2. Reconcile database discrepancies with uncertainty quantification
3. Predict phase stability as a function of external conditions
4. Build a unified, quality-controlled database

---

## 🔬 SCIENTIFIC BACKGROUND

### What is a Ground State?

The **ground state** is the most stable structure (lowest energy) of a material at given conditions (T, P).

For a composition like TiO₂:
- **Rutile** (space group 136): Ground state at ambient conditions
- **Anatase** (space group 141): Metastable polymorph (E_hull = 0.012 eV/atom)
- **Brookite** (space group 61): Another metastable form

### The Convex Hull

The **convex hull** determines thermodynamic stability:

```
Formation Energy (eV/atom)
   ↑
   │     Stable compounds (on hull)
   │     ●────●────●────●
   │    ╱      ╲  ╱  ╲
   │   ●        ●    ●  Unstable
   │  ╱              ╲  (above hull)
   │ ●                ●
   └─────────────────────→
     Composition (x in TixO(1-x))
```

- **On hull (E_hull = 0):** Thermodynamically stable
- **Above hull (E_hull > 0):** Unstable (will decompose)
- **Near hull (E_hull < 0.025):** Potentially synthesizable

### Why is This Computationally Expensive?

To find the ground state of a new composition:

1. **Test all 230 space groups** (or use structure prediction algorithms)
2. **For each space group:**
   - Generate atomic positions
   - Run DFT calculation (~1-10 CPU hours)
   - Relax structure
3. **Calculate competing phases** (all possible decomposition products)
4. **Construct convex hull**

**Total cost:** 1000-10,000 CPU hours per composition!

### Database Discrepancies

Same material, different databases:

| Property | Materials Project | OQMD | JARVIS | Why Different? |
|----------|------------------|------|--------|----------------|
| Fe₂O₃ volume | 101.2 Ų | 101.5 Ų | 99.8 Ų | PBE vs OptB88-vdW |
| Formation E | -2.51 eV | -2.48 eV | -2.53 eV | Pseudopotentials |
| Band gap | 2.2 eV | 2.0 eV | 2.1 eV | k-point density |

**This creates confusion:** Which value should researchers trust?

---

## 🚀 PROJECT APPROACH

### Core Innovation: Hierarchical ML-Guided Workflow

Instead of blindly testing all structures, we use ML to filter candidates:

```
INPUT: Composition (e.g., "Li₃FeO₄")
   ↓
┌──────────────────────────────────────┐
│  STAGE 1: ML PRE-SCREENING           │
│  ├─ Space group prediction           │
│  │  "Most likely: 227, 225, 141"     │
│  ├─ Formation energy estimation      │
│  │  "E_f ≈ -3.2 ± 0.4 eV/atom"      │
│  ├─ Hull distance prediction         │
│  │  "E_hull ≈ 0.015 ± 0.05"         │
│  └─ Volume prediction                │
│     "V ≈ 145 ± 8 Ų"                 │
└──────────────────────────────────────┘
   ↓
DECISION: Is E_hull < 0.05? → YES, proceed
   ↓
┌──────────────────────────────────────┐
│  STAGE 2: CONSTRAINED STRUCTURE      │
│            GENERATION                 │
│  Use USPEX/CALYPSO/PSO with:         │
│  - Only top-5 space groups           │
│  - Volume range from ML              │
│  - 300 structures (not 3000!)        │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  STAGE 3: RAPID DFT SCREENING        │
│  - PBE functional (fast)             │
│  - Moderate k-points                 │
│  - Validate ML predictions           │
│  - Keep top 5 candidates             │
└──────────────────────────────────────┘
   ↓
┌──────────────────────────────────────┐
│  STAGE 4: HIGH-ACCURACY DFT          │
│  - SCAN/r²SCAN functional            │
│  - Dense k-points                    │
│  - Phonon calculations               │
│  - Final hull construction           │
└──────────────────────────────────────┘
   ↓
OUTPUT: Unified database entry with uncertainties
```

**Key Advantages:**
- **10-100× faster** than traditional approach
- **Quality-controlled** through multi-stage validation
- **Uncertainty-aware** predictions
- **Database reconciliation** built-in

---

## 🧠 MACHINE LEARNING MODELS

### Model 1: Space Group Predictor

**Input:** Composition only (e.g., "Fe₂O₃")  
**Output:** Probability distribution over 230 space groups

**Architecture Options:**
- **CrabNet** (Composition-based transformer) — *Recommended*
- **Roost** (Message-passing on composition)
- **Random Forest** (baseline with Magpie features)

**Training Data:**
- Combine MP, OQMD, AFLOW, JARVIS (~1.5M structures)
- Handle polymorphs by predicting ground state
- Class imbalance: Some space groups very rare

**Features (Descriptors):**

```python
# 1. Elemental properties (weighted by stoichiometry)
features_elemental = {
    'mean_atomic_radius': 1.23 Å,
    'range_electronegativity': 1.8,
    'mean_ionization_energy': 8.4 eV,
    'total_valence_electrons': 24,
    'mean_atomic_mass': 55.8 amu,
    'std_covalent_radius': 0.4 Å
}

# 2. Composition-based
features_composition = {
    'n_elements': 2,  # Binary
    'stoichiometry': [0.4, 0.6],  # Fe₂O₃
    'mixing_entropy': 0.67,
    'size_mismatch': 0.34
}

# 3. Crystal chemistry
features_chemistry = {
    'radius_ratio': 0.72,  # r_cation/r_anion
    'ionic_character': 0.45,
    'tolerance_factor': 0.89,  # For perovskites
    'is_oxide': True,
    'contains_transition_metal': True
}

# 4. Historical/Statistical
features_historical = {
    'sg_frequency_this_chemistry': [0.23, 0.18, ...],  # For oxides
    'n_known_polymorphs_similar': 3,
    'prototype_similarity': 'corundum-like'
}
```

**Total features:** ~130 (using Matminer's Magpie preset)

**Evaluation Metrics:**
- Top-1 accuracy: "Did we predict the correct SG?"
- Top-5 accuracy: "Is correct SG in top 5?"
- Entropy of prediction: "How confident is the model?"

---

### Model 2: Formation Energy & Hull Distance Predictor

**Purpose:** Pre-screen which compositions are likely stable

**Architecture:** Multi-task learning
```python
model_outputs = {
    'formation_energy': regression,      # eV/atom
    'energy_above_hull': regression,     # eV/atom
    'is_stable': binary_classification   # E_hull < 0.025?
}
```

**Why Multi-task?**
- Shared representations improve generalization
- Formation energy helps predict hull distance

**Models to Try:**
1. **Roost/CrabNet** (composition-only) — Start here
2. **MEGNet** (if structure available)
3. **Ensemble** of above

**Key Challenge:** Functional correction
- Materials Project uses PBE
- OQMD uses PBE
- But energies still differ!

**Solution:** Learn functional corrections
```python
# Add "database source" as a feature
features['from_MP'] = 1 or 0
features['from_OQMD'] = 1 or 0

# Or: Explicitly model correction
E_corrected = E_raw + Σ(α_i × n_i)  # Per-element correction
```

---

### Model 3: Volume/Lattice Parameter Predictor

**Input:** Composition + predicted space group  
**Output:** Cell volume, lattice parameters

**Why Important?**
- Initial guess for USPEX/CALYPSO
- Reduces structure search space
- Validates DFT convergence

**Architecture:**
```python
# Separate models for different crystal systems
if space_group in [1, 2]:  # Triclinic
    predict: a, b, c, α, β, γ  # 6 parameters
elif space_group in [3, ..., 15]:  # Monoclinic
    predict: a, b, c, β  # 4 parameters
elif space_group in [...]:  # Cubic
    predict: a  # 1 parameter only!
```

**Features:**
- All previous features +
- `space_group_number` (one-hot encoded or embedding)
- `crystal_system` (triclinic/monoclinic/etc.)

---

### Model 4: Phase Transition Predictor (Advanced)

**Goal:** Predict which polymorph is stable at (T, P)

**Challenge:** Most data is at T=0K, P=0!

**Approach:**

**Option A: Transfer Learning (Simpler)**
1. Pre-train on DFT data (T=0K, P=0)
2. Fine-tune on experimental phase diagrams
3. Data sources:
   - NIMS AtomWork
   - ASM Phase Diagrams
   - ICSD with temperature info

**Option B: Physics-Informed ML (Advanced)**
```python
# Predict Gibbs free energy
G(T, P, structure) = H - T×S + P×V

# Where:
H = DFT enthalpy (from database)
S = entropy (from phonons or ML model)
V = volume (from DFT or ML)

# Phase transition: G_phase1(T,P) = G_phase2(T,P)
```

**Simplified MVP:** 
- Focus on polymorphs at ambient conditions
- Predict: "Is anatase or rutile stable at 300K?"
- Use phonon entropy from JARVIS-DFT

---

## 📊 DATA STRATEGY

### Data Collection

**Databases to Query:**

```python
databases = {
    'Materials Project': {
        'access': 'pymatgen MPRester API',
        'entries': 154_000,
        'functional': 'PBE',
        'key_data': ['structure', 'energy', 'hull', 'elasticity']
    },
    'OQMD': {
        'access': 'qmpy package or direct download',
        'entries': 1_000_000,
        'functional': 'PBE',
        'key_data': ['structure', 'energy', 'hull']
    },
    'AFLOW': {
        'access': 'AFLUX API',
        'entries': 3_700_000,
        'functional': 'PBE',
        'key_data': ['structure', 'energy', 'multiple properties']
    },
    'JARVIS': {
        'access': 'jarvis-tools Python package',
        'entries': 75_000,
        'functional': 'OptB88-vdW and PBE',
        'key_data': ['structure', 'energy', 'phonons', 'elastic', 'optoelectronic']
    }
}
```

**What to Extract:**

```python
required_fields = [
    # Structural
    'composition',
    'space_group_number',
    'space_group_symbol',
    'crystal_system',
    'volume',
    'lattice_parameters',  # a, b, c, α, β, γ
    'atomic_positions',
    
    # Energetic
    'formation_energy_per_atom',
    'energy_above_hull',
    'total_energy',
    
    # Computational details (for reconciliation!)
    'functional',  # PBE, LDA, SCAN, etc.
    'pseudopotential',  # PAW, ONCV, etc.
    'k_point_density',
    'energy_cutoff',
    'is_magnetic',
    'magnetic_ordering',
    
    # Metadata
    'database_source',
    'calculation_date',
    'database_version'
]
```

---

### Data Cleaning & Harmonization

**1. Remove Duplicates**
```python
# Same composition + space group from different databases
# Keep all, but mark as duplicates for uncertainty quantification
```

**2. Handle Polymorphs**
```python
# TiO₂ appears in multiple space groups
data['TiO2'] = {
    'ground_state': {'sg': 136, 'E_hull': 0.0},
    'polymorphs': [
        {'sg': 141, 'E_hull': 0.012},  # Anatase
        {'sg': 61, 'E_hull': 0.025}    # Brookite
    ]
}
```

**3. Functional Corrections**
```python
# Build correction model
# Method: Train on materials calculated with multiple functionals
from sklearn.linear_model import Ridge

# Features: elemental composition
# Target: E_PBE - E_SCAN (or other functional pairs)
correction_model = Ridge().fit(X_composition, y_energy_difference)

# Apply corrections
E_corrected = E_PBE + correction_model.predict(composition)
```

**4. Quality Filters**
```python
# Exclude calculations with:
filters = {
    'not_converged': remove if final forces > 0.05 eV/Å,
    'too_few_kpoints': remove if k-point density < 500/Å⁻¹,
    'suspicious_energy': remove if |E_f| > 20 eV/atom,
    'structure_issues': remove if atoms too close < 0.5 Å
}
```

---

### Train/Validation/Test Split

**Critical:** Must avoid data leakage!

```python
# Strategy 1: Random split by composition (NOT by entry)
compositions = unique(['TiO2', 'Fe2O3', ...])
train_comp, test_comp = train_test_split(compositions, test_size=0.15)

# Strategy 2: Chemical family split (more challenging)
test_comp = ['Materials with Sc', 'Materials with rare earths']
train_comp = ['Everything else']

# Strategy 3: Time-based split
train = ['Everything before 2023']
test = ['Entries from 2024-2025']
```

**Recommended split:**
- Training: 70% (compositions)
- Validation: 15% (for hyperparameter tuning)
- Test: 15% (held-out, report final metrics)

---

### Uncertainty Quantification

**Sources of Uncertainty:**

1. **Model uncertainty:** ML prediction variance
2. **Data uncertainty:** Discrepancies between databases
3. **Functional uncertainty:** PBE vs SCAN vs experiments

**Approach:**

```python
# For each material
material_entry = {
    'composition': 'Fe2O3',
    
    # Predicted values with uncertainties
    'space_group': {
        'predicted': 167,
        'confidence': 0.89,
        'top_5': [(167, 0.89), (148, 0.06), ...]
    },
    
    'formation_energy': {
        'predicted': -2.50,
        'ml_uncertainty': 0.08,  # From ensemble models
        'database_spread': 0.03,  # MP vs OQMD difference
        'total_uncertainty': 0.09  # Combined
    },
    
    'volume': {
        'predicted': 100.8,
        'uncertainty': 0.7,
        'sources': {
            'MP_PBE': 101.2,
            'OQMD_PBE': 101.5,
            'JARVIS_OptB88': 99.8
        }
    }
}
```

**Methods:**
- **Ensemble models:** Train 5-10 models, report std of predictions
- **Bayesian neural networks:** Built-in uncertainty
- **Bootstrapping:** Resample training data, measure variance

---

## 🔧 IMPLEMENTATION DETAILS

### Software Stack

```python
# Core libraries
numpy >= 1.24
pandas >= 2.0
scipy >= 1.10

# Materials science
pymatgen >= 2023.10.11  # Structure manipulation, MP API
ase >= 3.22  # Atomic simulation environment
jarvis-tools >= 2023.8.1  # JARVIS database
matminer >= 0.9.0  # Feature engineering

# Machine learning
scikit-learn >= 1.3
xgboost >= 1.7  # Baseline models
tensorflow >= 2.13  # Deep learning
pytorch >= 2.0  # Alternative to TF

# Specialized ML for materials
roost  # Composition-based NN
crabnet  # Composition transformer
megnet  # Graph NN for crystals

# DFT workflow
pymatgen >= 2023.10.11  # VASP/QE interface
atomate >= 1.0  # DFT workflow automation
fireworks >= 2.0  # Job management

# Visualization
matplotlib >= 3.7
seaborn >= 0.12
plotly >= 5.17

# Database
mongodb  # For storing results
sqlalchemy  # Alternative: SQL database
```

---

### Computing Requirements

**For ML Training:**
- **CPU:** 16-32 cores recommended
- **RAM:** 64-128 GB (for large datasets)
- **GPU:** 1x RTX 4090 or A100 (for deep learning models)
- **Storage:** 500 GB - 1 TB (for database downloads)
- **Time:** 1-2 weeks for full training pipeline

**For DFT Validation:**
- **CPU:** HPC cluster with 100-1000 cores
- **RAM:** 4-8 GB per core
- **Storage:** 10-50 TB (DFT outputs are large!)
- **Time:** Depends on number of validations
  - 100 materials × 5 structures = 500 DFT calculations
  - ~2-10 hours per calculation
  - Total: 1000-5000 CPU-hours

**Estimated Total Cost:**
- ML development: ~$500 (cloud computing)
- DFT validation: ~$5,000-20,000 (HPC time)

---

### Directory Structure

```
ml-materials-database/
│
├── data/
│   ├── raw/                    # Downloaded databases
│   │   ├── materials_project/
│   │   ├── oqmd/
│   │   ├── aflow/
│   │   └── jarvis/
│   ├── processed/              # Cleaned, harmonized data
│   │   ├── train.pkl
│   │   ├── val.pkl
│   │   └── test.pkl
│   └── features/               # Pre-computed features
│
├── models/
│   ├── space_group/            # SG prediction model
│   │   ├── config.yaml
│   │   ├── train.py
│   │   └── best_model.pt
│   ├── formation_energy/
│   ├── volume/
│   └── phase_transition/
│
├── dft/
│   ├── inputs/                 # VASP/QE input files
│   ├── outputs/                # DFT results
│   └── workflows/              # Automation scripts
│
├── database/
│   ├── schema.py               # Database design
│   ├── populate.py             # Insert data
│   └── query.py                # API for users
│
├── analysis/
│   ├── model_evaluation.ipynb
│   ├── database_comparison.ipynb
│   └── case_studies.ipynb
│
├── docs/
│   └── api_documentation.md
│
└── scripts/
    ├── 01_download_databases.py
    ├── 02_clean_data.py
    ├── 03_featurize.py
    ├── 04_train_models.py
    ├── 05_predict.py
    └── 06_validate_dft.py
```

---

## 📅 PROJECT TIMELINE (16 weeks)

### **Phase 1: Data Collection & Preparation (Weeks 1-3)**

**Week 1: Database Download**
- [ ] Set up accounts (Materials Project, JARVIS, etc.)
- [ ] Download MP data using MPRester API
- [ ] Download OQMD via qmpy or direct download
- [ ] Download JARVIS-DFT
- [ ] Download AFLOW subset (focus on ground states)

**Deliverable:** Raw data in standardized format

**Week 2: Data Cleaning**
- [ ] Merge all databases into single DataFrame
- [ ] Handle duplicates (same composition + SG)
- [ ] Identify polymorphs
- [ ] Remove low-quality entries
- [ ] Compute additional fields (e.g., prototype labels)

**Deliverable:** Clean dataset ready for featurization

**Week 3: Feature Engineering**
- [ ] Extract compositional features using Matminer
- [ ] Compute crystal chemistry descriptors
- [ ] Generate historical/statistical features
- [ ] Create train/val/test splits
- [ ] Save processed data

**Deliverable:** Feature matrices for ML training

---

### **Phase 2: ML Model Development (Weeks 4-8)**

**Week 4: Baseline Models**
- [ ] Train Random Forest for space group prediction
- [ ] Train XGBoost for formation energy
- [ ] Establish baseline metrics
- [ ] Error analysis: Where do simple models fail?

**Deliverable:** Baseline performance benchmarks

**Week 5-6: Advanced Models - Space Group**
- [ ] Implement CrabNet/Roost for SG prediction
- [ ] Hyperparameter tuning (learning rate, layers, etc.)
- [ ] Ensemble of models
- [ ] Uncertainty quantification (prediction entropy)

**Deliverable:** Trained SG prediction model with >80% top-5 accuracy

**Week 7-8: Advanced Models - Energy & Volume**
- [ ] Train formation energy predictor (multi-task with hull distance)
- [ ] Train volume predictor
- [ ] Cross-validation across databases
- [ ] Functional correction model

**Deliverable:** Energy predictor with MAE < 0.15 eV/atom

---

### **Phase 3: Integration & Testing (Weeks 9-10)**

**Week 9: End-to-End Pipeline**
- [ ] Build prediction pipeline: composition → SG, energy, volume
- [ ] Interface with structure generation (USPEX/CALYPSO)
- [ ] Automated DFT input generation
- [ ] Test on 20 known materials (sanity check)

**Deliverable:** Working ML → DFT pipeline

**Week 10: Validation Study**
- [ ] Select 50 test compositions (not in training data)
- [ ] Run ML predictions
- [ ] Generate structures with USPEX (constrained by ML)
- [ ] Compare: ML-guided vs random structure search

**Metric:** How much faster is ML-guided approach?

**Deliverable:** Validation results, compute savings analysis

---

### **Phase 4: DFT Validation (Weeks 11-13)**

**Week 11-12: Rapid DFT Screening**
- [ ] Run PBE calculations on ML-filtered structures
- [ ] Compare predicted vs DFT energies/volumes
- [ ] Identify failures, retrain models if needed
- [ ] Select top candidates for high-accuracy DFT

**Deliverable:** 200-500 rapid DFT calculations completed

**Week 13: High-Accuracy DFT**
- [ ] Run SCAN/r²SCAN on best candidates
- [ ] Phonon calculations for dynamic stability
- [ ] Construct accurate convex hulls
- [ ] Compare with database values

**Deliverable:** 50-100 high-quality DFT entries

---

### **Phase 5: Database Construction (Weeks 14-16)**

**Week 14: Database Design & Population**
- [ ] Design schema with uncertainty fields
- [ ] Set up MongoDB/PostgreSQL
- [ ] Populate with ML predictions + DFT validations
- [ ] Implement data provenance tracking

**Deliverable:** Working database with API

**Week 15: Analysis & Visualization**
- [ ] Compare our database with existing ones
- [ ] Case studies: specific material families
- [ ] Generate figures for publication
- [ ] Identify interesting materials for experimentalists

**Deliverable:** Analysis notebooks and figures

**Week 16: Documentation & Dissemination**
- [ ] Write API documentation
- [ ] Create tutorial notebooks
- [ ] Draft manuscript for publication
- [ ] Prepare presentation

**Deliverable:** Ready-to-publish database and paper

---

## 👥 TEAM ROLES & RESPONSIBILITIES

### **Team Structure (3-5 people)**

#### **Student 1: Data Engineer**
**Responsibilities:**
- Download and clean databases
- Build data processing pipeline
- Manage data storage and versioning
- Quality control

**Skills needed:**
- Python (pandas, numpy)
- Database management (SQL/MongoDB)
- Data cleaning experience

**Time commitment:** 15-20 hrs/week

---

#### **Student 2: ML Scientist**
**Responsibilities:**
- Implement ML models (CrabNet, Roost, etc.)
- Hyperparameter tuning
- Model evaluation and error analysis
- Uncertainty quantification

**Skills needed:**
- Machine learning (PyTorch/TensorFlow)
- Statistical analysis
- Materials science knowledge (helpful)

**Time commitment:** 20-25 hrs/week

---

#### **Student 3: Computational Chemist**
**Responsibilities:**
- Set up DFT workflows (VASP/Quantum Espresso)
- Run and monitor DFT calculations
- Analyze results and compare with ML
- Phonon and elastic property calculations

**Skills needed:**
- DFT experience (VASP/QE)
- HPC cluster management
- Materials science background

**Time commitment:** 20-25 hrs/week

---

#### **Student 4 (Optional): Database Developer**
**Responsibilities:**
- Design database schema
- Build web interface/API
- Create visualization dashboard
- User documentation

**Skills needed:**
- Web development (Flask/Django)
- Database design
- API development

**Time commitment:** 10-15 hrs/week

---

#### **Student 5 (Optional): Materials Scientist**
**Responsibilities:**
- Domain expertise on material families
- Case study selection and analysis
- Validation of predictions
- Manuscript writing

**Skills needed:**
- Materials science PhD-level knowledge
- Literature review
- Scientific writing

**Time commitment:** 10-15 hrs/week

---

## 💡 GETTING STARTED: FIRST STEPS

### Week 1 To-Do List

**For the Team Leader:**
1. Set up GitHub repository
2. Create Slack/Discord for communication
3. Schedule weekly meetings
4. Assign initial roles

**For Student 1 (Data Engineer):**
1. Register for Materials Project API key at https://next-gen.materialsproject.org/api
2. Install required packages:
```bash
pip install pymatgen mp-api jarvis-tools matminer
```
3. Download small test dataset (1000 materials) from MP
4. Explore data structure, identify key fields

**For Student 2 (ML Scientist):**
1. Literature review: Read papers on materials ML
   - "CrabNet: Attentive deep learning models for chemistry" (2021)
   - "Roost: Machine learning for materials" (2020)
2. Set up ML environment:
```bash
conda create -n materials-ml python=3.10
conda activate materials-ml
pip install torch scikit-learn wandb
```
3. Try example code from matminer tutorials

**For Student 3 (Computational Chemist):**
1. Test HPC cluster access
2. Run test VASP calculation on known material (e.g., Si)
3. Set up workflow management (Atomate or custom scripts)
4. Estimate compute time for different system sizes

---

### Initial Meeting Agenda (Week 1)

**Objectives:**
1. Introduce everyone, clarify roles
2. Review project goals and timeline
3. Discuss technical challenges
4. Set up infrastructure (GitHub, communication)

**Discussion Points:**
- Which databases should we prioritize?
- What ML models to start with?
- HPC resource allocation
- Publication strategy

**Deliverables:**
- Assigned roles
- Shared Google Drive/Dropbox
- GitHub repository
- Communication channel

---

### Key Resources

**Papers to Read:**
1. "Machine learning for materials scientists" - Nature Reviews (2019)
2. "The Materials Project: A materials genome approach" - APL Materials (2013)
3. "AFLOW: An automatic framework for high-throughput materials discovery" - Comp Mat Sci (2012)
4. "CrabNet: Attentive DL models for chemistry" - NPJ Comp Mat (2021)
5. "Uncertainty quantification in ML for materials" - Chem Rev (2021)

**Online Courses:**
- Materials Project workshop: https://workshop.materialsproject.org/
- Matminer tutorials: https://github.com/hackingmaterials/matminer_examples
- PyMatGen documentation: https://pymatgen.org/

**Community:**
- Materials Project forum: https://matsci.org/
- JARVIS repository: https://github.com/usnistgov/jarvis

---

## 🎯 SUCCESS METRICS

### Technical Milestones

**ML Model Performance:**
- [ ] Space group prediction: Top-5 accuracy > 80%
- [ ] Formation energy: MAE < 0.15 eV/atom
- [ ] Hull distance: MAE < 0.05 eV/atom
- [ ] Volume prediction: MAPE < 5%

**Computational Efficiency:**
- [ ] ML pre-screening reduces DFT calculations by 10-50×
- [ ] Tier-1 DFT validates >70% of ML predictions
- [ ] Total time: composition → validated ground state < 48 hours

**Database Quality:**
- [ ] 1000+ new materials with uncertainties
- [ ] Consistency checks: all entries validated by DFT
- [ ] Documentation: every field explained
- [ ] API: researchers can query programmatically

---

### Publication Targets

**Primary Paper:** 
*"Machine Learning-Guided Materials Discovery: A Hierarchical Approach to Ground State Prediction"*

**Target Journals:**
- NPJ Computational Materials (Impact Factor: 9.7)
- Chemistry of Materials (IF: 8.6)
- Journal of Chemical Information and Modeling (IF: 5.6)

**Secondary Papers:**
- Database paper in Scientific Data
- ML methodology in Machine Learning: Science and Technology

**Presentations:**
- Materials Research Society (MRS) Meeting
- American Physical Society (APS) March Meeting
- Machine Learning for Materials workshop

---

### Community Impact

**Metrics:**
- Database downloads/API calls
- Citations of published papers
- GitHub stars/forks
- Integration with other tools (ASE, AiiDA, etc.)

**Long-term Vision:**
- Standard tool for materials screening
- Integration into Materials Project ecosystem
- Extension to other property predictions

---

## ⚠️ POTENTIAL CHALLENGES & SOLUTIONS

### Challenge 1: Database Access Issues

**Problem:** Some databases require licenses or have rate limits

**Solutions:**
- Start with open databases (MP, JARVIS)
- For OQMD: Use bulk download option
- For AFLOW: Request academic access
- For ICSD: Access through university subscription

---

### Challenge 2: Model Overfitting

**Problem:** ML models memorize training data, poor generalization

**Solutions:**
- Strict train/test split by chemical family
- Cross-validation across databases
- Ensemble models to reduce variance
- Regularization (dropout, L2 penalty)

---

### Challenge 3: DFT Computational Cost

**Problem:** Even with ML filtering, DFT validation is expensive

**Solutions:**
- Prioritize validation: focus on interesting/novel predictions
- Use multi-fidelity approach: cheap DFT first, accurate later
- Collaborate: share compute resources with other groups
- Active learning: validate materials that reduce model uncertainty most

---

### Challenge 4: Database Discrepancies

**Problem:** Different databases give different values for same material

**Solutions:**
- Track all sources, report uncertainty ranges
- Build functional correction models
- When in doubt: run our own high-quality DFT
- Transparency: clearly document which value we chose and why

---

### Challenge 5: Phase Transition Prediction

**Problem:** Limited experimental data at different T, P

**Solutions:**
- Start simple: polymorphs at ambient conditions
- Use phonon-based entropy estimates
- Transfer learning from available data
- Collaborate with experimental groups for validation

---

## 🔮 FUTURE EXTENSIONS

### Short-term (6 months post-completion)

1. **Extend to more properties:**
   - Band gaps
   - Elastic moduli
   - Dielectric constants

2. **Improve phase transition predictions:**
   - Incorporate phonon databases
   - Temperature-dependent hulls

3. **Web interface:**
   - User-friendly query system
   - Visualization of convex hulls
   - API for programmatic access

---

### Medium-term (1-2 years)

1. **Active learning loop:**
   - Model suggests most uncertain materials
   - DFT validates → model improves
   - Continuous improvement

2. **Inverse design:**
   - "Find me materials with E_hull < 0 and band gap = 2 eV"
   - Generative models for novel compositions

3. **Integration with experiments:**
   - Partner with synthesis groups
   - Validate predictions experimentally
   - Feedback loop

---

### Long-term (3-5 years)

1. **Universal materials predictor:**
   - One model for all properties
   - Foundation model for materials science

2. **Automated materials discovery:**
   - Closed-loop: ML → DFT → synthesis → characterization
   - Autonomous lab integration

3. **Community platform:**
   - Open-source, community-maintained database
   - Crowdsourced DFT calculations
   - Integration with Materials Project, NOMAD

---

## 📚 EXPECTED DELIVERABLES

### Software

1. **ML Models** (GitHub repository)
   - Trained models with weights
   - Inference code
   - Example notebooks

2. **Database** (Public release)
   - Hosted on cloud (AWS/Azure)
   - RESTful API
   - Python client library

3. **Documentation**
   - User guide
   - API reference
   - Tutorial notebooks

---

### Publications

1. **Main paper:** Methodology and results
2. **Database paper:** Description of database
3. **Case studies:** Applications to specific material families

---

### Presentations

1. Conference talks (MRS, APS)
2. Departmental seminars
3. Workshop tutorials

---

## 💰 BUDGET ESTIMATE

### Computing Resources

| Item | Cost | Notes |
|------|------|-------|
| Cloud GPU (ML training) | $500 | 1 month on AWS/Azure |
| HPC cluster (DFT) | $5,000 - $20,000 | Depends on # validations |
| Database hosting | $100/month | For public release |

**Total computing:** $10,000 - $25,000

---

### Software Licenses

| Item | Cost | Notes |
|------|------|-------|
| VASP license | University | Usually available |
| Materials databases | Free - $5,000 | MP/JARVIS free, ICSD paid |
| ML frameworks | Free | PyTorch, TensorFlow |

**Total software:** $0 - $5,000

---

### Other Expenses

| Item | Cost | Notes |
|------|------|-------|
| Conference travel | $2,000/person | For presentations |
| Publication fees | $0 - $3,000 | Open access optional |

**Total other:** $2,000 - $5,000

---

**Grand Total:** $12,000 - $35,000

*Note: Many universities provide HPC access and software licenses, reducing actual out-of-pocket costs*

---

## 📞 CONTACT & COLLABORATION

### Looking for Collaborators?

This project is ambitious and could benefit from:

- **Experimentalists:** To validate predictions
- **Other computational groups:** Share compute resources
- **ML experts:** Advanced model development
- **Database developers:** Professional-grade infrastructure

---

### How to Get Involved

**If you're interested in this project:**

1. **As a student:**
   - Contact project lead
   - Review prerequisites
   - Join for specific phases

2. **As a collaborator:**
   - Offer compute resources
   - Provide experimental validation
   - Contribute specific expertise

3. **As a user:**
   - Try our database when released
   - Provide feedback
   - Suggest new features

---

## 🎓 LEARNING OUTCOMES

**By the end of this project, students will:**

### Technical Skills

- [ ] Master materials databases (MP, OQMD, JARVIS, AFLOW)
- [ ] Develop ML models for materials (CrabNet, Roost, GNNs)
- [ ] Run and analyze DFT calculations (VASP/Quantum Espresso)
- [ ] Build and deploy databases (MongoDB/PostgreSQL)
- [ ] Create APIs and web interfaces

### Scientific Skills

- [ ] Understand thermodynamic stability (convex hulls)
- [ ] Know limitations of DFT (functionals, convergence)
- [ ] Critically evaluate materials predictions
- [ ] Reconcile discrepancies across data sources

### Professional Skills

- [ ] Collaborate in multidisciplinary teams
- [ ] Write scientific papers and proposals
- [ ] Present at conferences
- [ ] Manage large computational projects
- [ ] Version control and reproducible research

---

## 🏆 WHY THIS PROJECT MATTERS

### Scientific Impact

- **Accelerate materials discovery** from years → months
- **Unified database** resolves cross-database inconsistencies
- **New methodology** applicable beyond materials (molecules, catalysts, etc.)

### Technological Impact

- **Better batteries:** Faster discovery of electrolytes, cathodes
- **Efficient solar cells:** Optimal bandgap materials
- **Quantum computing:** Topological materials prediction
- **Catalysis:** Active and stable catalyst discovery

### Educational Impact

- **Train next generation** of materials data scientists
- **Open-source tools** for the community
- **Reproducible science:** All code and data public

---

## 🚀 READY TO START?

### Immediate Next Steps

1. **Assemble team:** Recruit 3-5 motivated students
2. **Secure resources:** HPC access, software licenses
3. **Literature review:** Read key papers (provided in References)
4. **Download test data:** Get familiar with databases (Week 1)
5. **First meeting:** Align on goals, timeline, roles

---

### Questions to Discuss in First Meeting

1. Which material families should we focus on?
2. Which databases have we accessed?
3. What ML frameworks are we comfortable with?
4. What HPC resources are available?
5. What's our publication timeline?

---

## 📖 REFERENCES

### Key Papers

1. **Materials Databases:**
   - Jain et al., "Commentary: The Materials Project" APL Materials (2013)
   - Curtarolo et al., "AFLOW: An automatic framework" Comp Mat Sci (2012)
   - Choudhary et al., "The joint automated repository for various integrated simulations (JARVIS)" NPJ Comp Mat (2020)

2. **Machine Learning for Materials:**
   - Goodall & Lee, "Predicting materials properties without crystal structure" Nat Comms (2020) [Roost]
   - Wang et al., "Compositionally restricted attention-based network" NPJ Comp Mat (2021) [CrabNet]
   - Xie & Grossman, "Crystal graph convolutional neural networks" PRL (2018) [CGCNN]

3. **Structure Prediction:**
   - Oganov & Glass, "Crystal structure prediction using USPEX" J Chem Phys (2006)
   - Wang et al., "CALYPSO: A method for crystal structure prediction" Comp Phys Comm (2012)

4. **Thermodynamic Stability:**
   - Hautier et al., "Finding nature's missing ternary oxide compounds" Chem Mater (2010)
   - Sun et al., "The thermodynamic scale of inorganic crystalline metastability" Science Advances (2016)

5. **Uncertainty Quantification:**
   - Scalia et al., "Evaluating scalable uncertainty estimation methods" Chem Sci (2020)
   - Dunn et al., "Benchmarking materials property prediction methods" NPJ Comp Mat (2020)

---

### Online Resources

- **Materials Project:** https://materialsproject.org/
- **JARVIS:** https://jarvis.nist.gov/
- **AFLOW:** http://aflowlib.org/
- **Matminer tutorials:** https://github.com/hackingmaterials/matminer_examples
- **PyMatGen docs:** https://pymatgen.org/

---

## 📝 APPENDIX

### A. Glossary

**DFT (Density Functional Theory):** Quantum mechanical method for calculating material properties

**Convex Hull:** Graph showing the most stable compounds for a given composition

**Space Group:** Symmetry classification for crystal structures (230 total)

**Formation Energy:** Energy released when forming a compound from elements

**E_hull (Energy Above Hull):** Distance from convex hull; stability measure

**Polymorph:** Different crystal structure of same composition (e.g., diamond vs graphite)

**Functional:** Approximation in DFT (PBE, LDA, SCAN, etc.)

**k-points:** Sampling of reciprocal space in DFT calculations

**Pseudopotential:** Approximation of core electrons in DFT

---

### B. Acronyms

- **MP:** Materials Project
- **OQMD:** Open Quantum Materials Database
- **JARVIS:** Joint Automated Repository for Various Integrated Simulations
- **DFT:** Density Functional Theory
- **ML:** Machine Learning
- **GNN:** Graph Neural Network
- **MAE:** Mean Absolute Error
- **MAPE:** Mean Absolute Percentage Error
- **HPC:** High-Performance Computing
- **API:** Application Programming Interface

---

### C. Code Templates

**Example: Downloading from Materials Project**

```python
from mp_api.client import MPRester

# Get API key from https://next-gen.materialsproject.org/api
with MPRester("YOUR_API_KEY") as mpr:
    # Search for all Fe-O compounds
    docs = mpr.materials.summary.search(
        chemsys="Fe-O",
        fields=["material_id", "formula_pretty", "energy_above_hull", 
                "formation_energy_per_atom", "structure"]
    )
    
    print(f"Found {len(docs)} materials")
    for doc in docs[:5]:
        print(f"{doc.formula_pretty}: E_hull = {doc.energy_above_hull} eV/atom")
```

**Example: Feature Engineering with Matminer**

```python
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

# Initialize featurizer
ep = ElementProperty.from_preset("magpie")

# Featurize a composition
comp = Composition("Fe2O3")
features = ep.featurize(comp)

print(f"Generated {len(features)} features")
print(f"Feature names: {ep.feature_labels()}")
```

**Example: Simple ML Model**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data
X = feature_matrix  # Shape: (n_materials, n_features)
y = space_groups    # Shape: (n_materials,)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Train
model = RandomForestClassifier(n_estimators=100, max_depth=20)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")

# Get feature importance
importance = model.feature_importances_
```

---

## ✅ PROJECT CHECKLIST

Use this to track progress:

### Setup Phase
- [ ] Team assembled and roles assigned
- [ ] GitHub repository created
- [ ] Communication channels established (Slack/Discord)
- [ ] HPC access confirmed
- [ ] Software licenses obtained

### Data Phase
- [ ] Materials Project data downloaded
- [ ] OQMD data downloaded
- [ ] JARVIS data downloaded
- [ ] AFLOW data downloaded
- [ ] Data cleaned and merged
- [ ] Features computed
- [ ] Train/val/test splits created

### ML Phase
- [ ] Baseline models trained (RF, XGBoost)
- [ ] Space group predictor trained
- [ ] Formation energy predictor trained
- [ ] Volume predictor trained
- [ ] Models evaluated on test set
- [ ] Uncertainty quantification implemented

### DFT Phase
- [ ] Test DFT calculations completed
- [ ] Rapid screening workflow established
- [ ] High-accuracy workflow established
- [ ] 50+ materials validated
- [ ] Results compared with ML predictions

### Database Phase
- [ ] Database schema designed
- [ ] Database populated
- [ ] API implemented
- [ ] Documentation written
- [ ] Web interface created (optional)

### Dissemination Phase
- [ ] Manuscript drafted
- [ ] Figures and tables prepared
- [ ] Submitted to journal
- [ ] Conference presentation prepared
- [ ] Code and data released publicly

---

## 🎉 CONCLUSION

This project represents a unique opportunity to:
- Develop cutting-edge ML methods for materials science
- Build a valuable resource for the research community
- Gain experience across multiple disciplines
- Publish high-impact papers

**The key innovation:** Hierarchical ML-guided approach that drastically reduces computational cost while maintaining accuracy through multi-stage validation.

**The expected impact:** Enable rapid materials discovery and provide a unified, quality-controlled database with uncertainty quantification.

---

**Ready to revolutionize materials discovery? Let's get started! 🚀**

---

*Document Version: 1.0*  
*Last Updated: October 31, 2025*  
*Contact: [Your Email/Institution]*

---

## NEXT STEPS FOR YOUR TEAM

1. **Read this proposal carefully** (everyone)
2. **Schedule kick-off meeting** (Week 1)
3. **Assign roles** based on expertise
4. **Set up infrastructure** (GitHub, Slack, HPC)
5. **Start with data collection** (Week 1-2)
6. **Check in weekly** to track progress

**Questions? Concerns? Ideas?** 
Bring them to the first meeting!

---

*Good luck, and happy materials discovery!* 🔬⚗️🧪
