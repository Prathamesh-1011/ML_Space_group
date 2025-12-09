# ML FEATURE/DESCRIPTOR REFERENCE GUIDE
## Comprehensive Feature Engineering for Materials Property Prediction

---

## OVERVIEW

This document provides detailed guidance on selecting and engineering features (descriptors) for machine learning models in materials science. This complements the main proposal by providing actionable feature engineering strategies.

---

## 🎯 WHY DESCRIPTORS MATTER

**Problem:** Raw composition string "Fe₂O₃" contains no numerical information for ML models.

**Solution:** Convert to meaningful numerical descriptors that capture:
- Chemical properties
- Electronic structure
- Crystal chemistry principles
- Historical patterns

**Key Principle:** Good features encode domain knowledge → better predictions with less data!

---

## 📊 DESCRIPTOR CATEGORIES

### 1. ELEMENTAL PROPERTIES

These are properties of individual elements, which we aggregate for compositions.

#### A. Basic Properties

```python
elemental_features = {
    # Atomic structure
    'atomic_number': [1-118],           # Z
    'atomic_mass': [1-300] amu,         # M
    'period': [1-7],                    # Row in periodic table
    'group': [1-18],                    # Column in periodic table
    'block': ['s', 'p', 'd', 'f'],      # Electronic block
    
    # Size properties
    'atomic_radius': Å,                 # Atomic radius
    'covalent_radius': Å,               # Covalent radius
    'ionic_radius': Å,                  # For common oxidation states
    'van_der_waals_radius': Å,          # VdW radius
}
```

**Where to get:**
- `pymatgen.core.periodic_table.Element`
- `mendeleev` Python package
- Matminer's `ElementProperty` featurizer

#### B. Electronic Structure

```python
electronic_features = {
    # Valence electrons
    'valence_electrons': [1-8],         # Number of valence e⁻
    'unfilled_d': [0-10],               # d-electrons
    'unfilled_f': [0-14],               # f-electrons
    
    # Energies
    'first_ionization_energy': eV,      # IE₁
    'second_ionization_energy': eV,     # IE₂
    'electron_affinity': eV,            # EA
    
    # Orbital info
    'valence_orbitals': ['s', 'p', 'd', 'f'],
    's_orbital_electrons': [0-2],
    'p_orbital_electrons': [0-6],
    'd_orbital_electrons': [0-10],
}
```

**Why important:**
- Transition metals (d-electrons) → magnetic properties
- f-electrons → rare earth magnetism
- Ionization energy → bond strength estimation

#### C. Chemical Properties

```python
chemical_features = {
    # Electronegativity (multiple scales available)
    'pauling_electronegativity': [0.7-4.0],     # Most common
    'allen_electronegativity': [0.7-3.5],       # Alternative
    'mulliken_electronegativity': varies,
    
    # Oxidation states
    'common_oxidation_states': [list],          # [+2, +3] for Fe
    'max_oxidation_state': int,
    'min_oxidation_state': int,
    
    # Classification
    'is_metal': bool,
    'is_metalloid': bool,
    'is_nonmetal': bool,
    'is_noble_gas': bool,
    'is_transition_metal': bool,
    'is_lanthanoid': bool,
    'is_actinoid': bool,
}
```

**Implementation:**
```python
from pymatgen.core import Element

fe = Element("Fe")
print(f"Electronegativity: {fe.X}")           # Pauling scale
print(f"Oxidation states: {fe.common_oxidation_states}")
print(f"Is transition metal: {fe.is_transition_metal}")
```

---

### 2. COMPOSITION-BASED DESCRIPTORS

These describe the overall composition, not individual elements.

#### A. Stoichiometry Features

```python
from pymatgen.core import Composition

comp = Composition("Fe2O3")

stoich_features = {
    # Basic
    'n_elements': 2,                            # Binary, ternary, etc.
    'total_atoms': 5,                           # Fe₂O₃ has 5 atoms
    
    # Fractions
    'element_fractions': {
        'Fe': 0.4,  # 2/5
        'O': 0.6    # 3/5
    },
    
    # Atomic vs weight fractions
    'atomic_fraction_Fe': 0.4,
    'weight_fraction_Fe': 0.699,               # Mass-weighted
    
    # Complexity
    'is_binary': True,
    'is_ternary': False,
    'is_quaternary': False,
}
```

#### B. Weighted Averages

Most powerful! Combine stoichiometry with elemental properties:

```python
# Formula: mean_property = Σ(fraction_i × property_i)

weighted_features = {
    # Averages
    'mean_atomic_mass': Σ(f_i × M_i),
    'mean_atomic_radius': Σ(f_i × r_i),
    'mean_electronegativity': Σ(f_i × χ_i),
    'mean_ionization_energy': Σ(f_i × IE_i),
    'mean_electron_affinity': Σ(f_i × EA_i),
    
    # Ranges (capture heterogeneity)
    'range_electronegativity': max(χ) - min(χ),
    'range_atomic_radius': max(r) - min(r),
    'range_ionization_energy': max(IE) - min(IE),
    
    # Standard deviations
    'std_atomic_mass': σ([M_1, M_2, ...]),
    'std_atomic_radius': σ([r_1, r_2, ...]),
    
    # Maximum/minimum
    'max_electronegativity': max(χ_i),
    'min_electronegativity': min(χ_i),
}
```

**Example calculation:**
```python
comp = Composition("Fe2O3")
elements = comp.elements
fractions = comp.fractional_composition.values()

# Mean electronegativity
mean_chi = sum(f * el.X for f, el in zip(fractions, elements))
# For Fe₂O₃: 0.4 × 1.83 + 0.6 × 3.44 = 2.80

# Range
chi_values = [el.X for el in elements]
range_chi = max(chi_values) - min(chi_values)
# For Fe₂O₃: 3.44 - 1.83 = 1.61
```

#### C. Mixing Properties

```python
mixing_features = {
    # Entropy of mixing (configurational)
    'mixing_entropy': -Σ(f_i × ln(f_i)),       # Higher → more disorder
    
    # Size mismatch (Hume-Rothery rules)
    'size_mismatch': σ²([r_1, r_2, ...]),      # Variance of radii
    
    # Mass variance
    'mass_variance': σ²([M_1, M_2, ...]),
    
    # Heterogeneity
    'composition_heterogeneity': geometric_mean / arithmetic_mean,
}
```

**Why mixing entropy matters:**
- High entropy → solid solutions possible
- Low entropy → ordered compounds preferred
- Correlates with phase stability

---

### 3. CRYSTAL CHEMISTRY DESCRIPTORS

These encode chemical intuition and rules.

#### A. Ionic Interactions

For ionic compounds (oxides, halides, etc.):

```python
ionic_features = {
    # Ionic character (Pauling)
    'ionic_character': 1 - exp(-0.25 × (χ_A - χ_B)²),
    
    # Alternative: Simple difference
    'chi_difference': |χ_cation - χ_anion|,
    
    # For Fe₂O₃
    'ionic_character': 1 - exp(-0.25 × (3.44-1.83)²) = 0.57,
    
    # Estimated lattice energy (Born-Landé)
    'madelung_estimate': q₁ × q₂ / (r₁ + r₂),
}
```

**Physical meaning:**
- Ionic character > 0.5 → predominantly ionic bonding
- Ionic character < 0.3 → covalent bonding

#### B. Radius Ratios

Predict coordination environments:

```python
radius_ratio_features = {
    # Basic ratio
    'radius_ratio': r_cation / r_anion,
    
    # Pauling's rules predictions:
    # r/R < 0.155 → 2-coordinate (linear)
    # 0.155-0.225 → 3-coordinate (trigonal)
    # 0.225-0.414 → 4-coordinate (tetrahedral)
    # 0.414-0.732 → 6-coordinate (octahedral)
    # 0.732-1.0 → 8-coordinate (cubic)
    
    'predicted_coordination': derived_from_above,
}
```

**Example:**
```python
# For Fe²⁺ in oxides
r_Fe2 = 0.78  # Å (high-spin octahedral)
r_O = 1.40    # Å

ratio = 0.78 / 1.40 = 0.56
# Predicts: octahedral coordination ✓ (matches reality!)
```

#### C. Goldschmidt Tolerance Factor

For perovskite structures (ABX₃):

```python
tolerance_factor = {
    'formula': (r_A + r_X) / [√2 × (r_B + r_X)],
    
    # Interpretations:
    # t < 0.71 → ilmenite structure
    # 0.71-0.9 → orthorhombic perovskite
    # 0.9-1.0 → cubic perovskite
    # t > 1.0 → hexagonal structure
}
```

**Application:** Predicting perovskite stability for solar cells, catalysts

#### D. Chemical Intuition Features

```python
chemistry_boolean = {
    # Element type interactions
    'is_metal_metal': both elements metallic,
    'is_metal_nonmetal': one metal, one nonmetal,
    'is_nonmetal_nonmetal': both nonmetals,
    
    # Specific chemistries
    'contains_oxygen': bool,
    'contains_nitrogen': bool,
    'contains_fluorine': bool,
    'contains_carbon': bool,
    
    # Functional groups
    'is_oxide': True if contains O,
    'is_halide': True if contains F/Cl/Br/I,
    'is_chalcogenide': True if contains S/Se/Te,
    'is_nitride': True if contains N,
    
    # Element categories
    'contains_alkali': bool,
    'contains_alkaline_earth': bool,
    'contains_transition_metal': bool,
    'contains_rare_earth': bool,
    'contains_noble_gas': bool,
    
    # Counts
    'n_transition_metals': int,
    'n_p_block_elements': int,
}
```

---

### 4. HISTORICAL/STATISTICAL DESCRIPTORS

Learn from existing databases:

#### A. Space Group Frequency

```python
# For each composition, compute:
historical_sg = {
    # How often does this SG appear for similar compositions?
    'sg_frequency_oxides': P(SG=167 | binary oxide),
    'sg_frequency_transition_metal_oxides': P(SG=167 | TM oxide),
    
    # Weighted by chemical similarity
    'weighted_sg_frequency': Σ(similarity_i × P(SG|comp_i)),
}
```

**Implementation:**
```python
# Build lookup table from training data
sg_freq_oxides = training_data[
    training_data['contains_oxygen']
].groupby('space_group').size() / len(training_data)

# For new composition
if 'O' in composition:
    features['sg_freq'] = sg_freq_oxides
```

#### B. Prototype Structure Similarity

```python
prototype_features = {
    # Distance to known structure types
    'similarity_to_rocksalt': cosine_similarity(comp, NaCl),
    'similarity_to_wurtzite': cosine_similarity(comp, ZnS),
    'similarity_to_perovskite': cosine_similarity(comp, CaTiO3),
    'similarity_to_spinel': cosine_similarity(comp, MgAl2O4),
    'similarity_to_corundum': cosine_similarity(comp, Al2O3),
    
    # Closest prototype
    'closest_prototype': 'corundum',
    'prototype_distance': 0.15,
}
```

**Computing similarity:**
```python
from sklearn.metrics.pairwise import cosine_similarity

# Represent compositions as feature vectors
feat_new = featurize("Fe2O3")
feat_corundum = featurize("Al2O3")  # Prototype

similarity = cosine_similarity([feat_new], [feat_corundum])[0, 0]
```

#### C. Database Statistics

```python
database_features = {
    # How many polymorphs are known?
    'n_known_polymorphs': 3,  # For TiO₂
    
    # Typical stability
    'mean_hull_distance_chemistry': avg(E_hull for similar comps),
    'std_hull_distance_chemistry': std(E_hull for similar comps),
    
    # Prevalence
    'n_materials_this_chemistry': count(binary oxides in DB),
    'fraction_stable': P(E_hull = 0 | binary oxide),
}
```

---

### 5. GRAPH/NETWORK DESCRIPTORS

If you have crystal structure (not just composition):

#### A. Coordination Environment

```python
from pymatgen.analysis.local_env import CrystalNN

coordination_features = {
    # Per-site
    'coordination_number': [6, 4, ...],         # Each atom
    'mean_coordination': 5.2,
    'std_coordination': 0.8,
    
    # Polyhedral info
    'octahedral_sites': 2,
    'tetrahedral_sites': 1,
    
    # Coordination types
    'has_square_planar': bool,
    'has_octahedral': bool,
    'has_tetrahedral': bool,
}
```

#### B. Bond Properties

```python
bond_features = {
    # Distances
    'mean_bond_length': Å,
    'min_bond_length': Å,
    'max_bond_length': Å,
    'std_bond_length': Å,
    
    # Angles
    'mean_bond_angle': degrees,
    'std_bond_angle': degrees,
    
    # Specific bonds
    'mean_metal_oxygen_distance': Å,
    'mean_cation_anion_distance': Å,
}
```

#### C. Packing & Symmetry

```python
structure_features = {
    # Packing
    'packing_fraction': V_atoms / V_cell,
    'density': g/cm³,
    'volume_per_atom': Å³,
    
    # Symmetry
    'point_group': '432',
    'crystal_system': 'cubic',
    'space_group_number': 225,
    
    # Anisotropy
    'lattice_anisotropy': (a-c)/a for tetragonal,
    'volume_anisotropy': V/V_ideal_sphere,
}
```

---

## 🔧 IMPLEMENTATION GUIDE

### Using Matminer (Recommended)

Matminer provides pre-built featurizers:

```python
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
    IonProperty,
    ElementFraction
)

# Option 1: Use preset (easiest)
ep = ElementProperty.from_preset("magpie")
features = ep.featurize(comp)
# Returns 132 features!

# Option 2: Custom selection
ep = ElementProperty(
    features=['atomic_mass', 'X', 'atomic_radius'],
    stats=['mean', 'range', 'std']
)
features = ep.featurize(comp)
```

**Matminer presets:**
- `"magpie"`: 132 features (comprehensive)
- `"deml"`: 39 features (fast)
- `"matminer"`: 56 features (balanced)
- `"matscholar_el"`: 89 features (literature-derived)

### Complete Feature Pipeline

```python
import pandas as pd
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from matminer.featurizers.conversions import StrToComposition

# Load data
df = pd.DataFrame({'composition': ['Fe2O3', 'TiO2', 'SiO2']})

# Convert strings to Composition objects
str2comp = StrToComposition()
df = str2comp.featurize_dataframe(df, 'composition')

# Featurize
ep = ElementProperty.from_preset("magpie")
df = ep.featurize_dataframe(df, 'composition')

stoich = Stoichiometry()
df = stoich.featurize_dataframe(df, 'composition')

print(f"Total features: {len(df.columns)}")
```

---

## 📋 FEATURE SELECTION STRATEGY

### By Model Type

#### For Space Group Prediction (Composition Only)

**Essential features:**
```python
features_sg_prediction = [
    # Stoichiometry (10 features)
    'n_elements', 'stoichiometry_*',
    
    # Weighted elemental (30 features)
    'mean_*', 'range_*', 'std_*',  # For atomic_radius, X, IE, etc.
    
    # Crystal chemistry (10 features)
    'radius_ratio', 'ionic_character', 'tolerance_factor',
    
    # Historical (20 features)
    'sg_frequency_*', 'prototype_similarity_*',
    
    # Boolean chemistry (10 features)
    'is_oxide', 'contains_transition_metal', etc.
]
# Total: ~80 features
```

#### For Formation Energy Prediction

**Essential features:**
```python
features_energy_prediction = [
    # All compositional features +
    
    # Energetic (20 features)
    'cohesive_energy_estimate',
    'mean_ionization_energy',
    'pauling_electronegativity_difference',
    
    # If structure known (15 features)
    'coordination_numbers',
    'mean_bond_length',
    'space_group_number',
    'volume_per_atom',
]
# Total: ~100 features
```

#### For Volume Prediction

**Essential features:**
```python
features_volume_prediction = [
    # Compositional features +
    
    # Size-related (emphasis!)
    'mean_atomic_radius', 'range_atomic_radius',
    'mean_covalent_radius', 'mean_ionic_radius',
    
    # Space group (constrains volume)
    'space_group_number',
    'crystal_system',
    
    # Historical
    'mean_volume_similar_compositions',
]
# Total: ~60 features
```

---

## ⚖️ FEATURE IMPORTANCE ANALYSIS

After training, always check feature importance:

```python
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Get importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot top 20
plt.figure(figsize=(10, 8))
plt.barh(importance.head(20)['feature'], importance.head(20)['importance'])
plt.xlabel('Importance')
plt.title('Top 20 Features')
plt.tight_layout()
plt.show()

# Print top features
print(importance.head(20))
```

**What to look for:**
- Physical interpretability: Do top features make sense?
- Redundancy: Are similar features all important?
- Surprises: Unexpected important features → new insights!

---

## 🎯 FEATURE ENGINEERING BEST PRACTICES

### 1. Domain Knowledge First

❌ **Bad:** Throw all 500 features at the model
✅ **Good:** Select ~80-100 physically meaningful features

### 2. Handle Missing Values

```python
# Some elements lack certain properties
df['electron_affinity'].fillna(df['electron_affinity'].median(), inplace=True)

# Or: Use -999 as "missing" indicator
df['electron_affinity'].fillna(-999, inplace=True)
```

### 3. Normalize/Scale Features

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Why:** Different features have different scales (atomic radius ~1 Å, atomic mass ~100 amu)

### 4. Check for Multicollinearity

```python
import seaborn as sns

# Correlation matrix
corr = df[feature_columns].corr()

# Heatmap
sns.heatmap(corr, cmap='coolwarm', center=0)

# Remove highly correlated features (|r| > 0.95)
to_remove = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.95:
            to_remove.append(corr.columns[j])

df.drop(columns=to_remove, inplace=True)
```

### 5. Create Interaction Features (Carefully!)

```python
# Physical interactions
df['chi_diff_times_radius_ratio'] = df['chi_difference'] * df['radius_ratio']

# Don't create too many! Feature explosion.
```

---

## 📊 FEATURE COMPARISON TABLE

| Feature Type | # Features | Importance | When to Use | Computation Cost |
|-------------|-----------|-----------|------------|-----------------|
| Elemental (weighted) | 30-50 | ⭐⭐⭐⭐⭐ | Always | Low |
| Stoichiometry | 10-20 | ⭐⭐⭐⭐ | Always | Very Low |
| Crystal chemistry | 10-15 | ⭐⭐⭐⭐ | For stability, SG | Low |
| Historical | 20-30 | ⭐⭐⭐⭐ | If database available | Medium |
| Graph/structural | 20-40 | ⭐⭐⭐⭐⭐ | If structure known | High |
| Custom physics | 5-10 | ⭐⭐⭐ | Domain-specific | Medium |

---

## 🚀 QUICK START TEMPLATE

```python
"""
Complete feature engineering pipeline for materials
"""

import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty, Stoichiometry

def featurize_materials(compositions):
    """
    Args:
        compositions: List of composition strings
    Returns:
        DataFrame with features
    """
    df = pd.DataFrame({'composition_str': compositions})
    
    # Convert to Composition objects
    df['composition'] = df['composition_str'].apply(Composition)
    
    # 1. Elemental properties (Magpie)
    ep = ElementProperty.from_preset("magpie")
    ep_features = ep.featurize_many(df['composition'].tolist())
    df_ep = pd.DataFrame(ep_features, columns=ep.feature_labels())
    
    # 2. Stoichiometry
    stoich = Stoichiometry()
    stoich_features = stoich.featurize_many(df['composition'].tolist())
    df_stoich = pd.DataFrame(stoich_features, columns=stoich.feature_labels())
    
    # 3. Custom features
    df['n_elements'] = df['composition'].apply(lambda x: len(x.elements))
    df['contains_oxygen'] = df['composition'].apply(lambda x: 'O' in x)
    df['contains_tm'] = df['composition'].apply(
        lambda x: any(el.is_transition_metal for el in x.elements)
    )
    
    # Combine
    df_final = pd.concat([df, df_ep, df_stoich], axis=1)
    
    return df_final

# Usage
materials = ['Fe2O3', 'TiO2', 'SiO2', 'CaTiO3']
features_df = featurize_materials(materials)
print(f"Generated {len(features_df.columns)} features")
```

---

## 📚 RESOURCES

### Documentation
- **Matminer:** https://hackingmaterials.lbl.gov/matminer/
- **PyMatGen:** https://pymatgen.org/
- **Mendeleev package:** https://mendeleev.readthedocs.io/

### Papers
1. Ward et al., "A general-purpose machine learning framework for predicting properties of inorganic materials" *NPJ Comp Mat* (2016) [Magpie features]
2. Deml et al., "Predicting density functional theory total energies and enthalpies of formation" *PRB* (2016) [OQMD features]
3. Jha et al., "ElemNet: Deep learning the chemistry" *Sci Rep* (2018)

---

## ✅ FEATURE CHECKLIST

Use this when setting up your pipeline:

- [ ] Installed matminer and dependencies
- [ ] Loaded composition data
- [ ] Converted strings to Composition objects
- [ ] Applied Magpie/Deml preset
- [ ] Added stoichiometry features
- [ ] Added custom chemistry features
- [ ] Added historical features (if database available)
- [ ] Checked for missing values
- [ ] Scaled/normalized features
- [ ] Checked for multicollinearity
- [ ] Saved feature matrix for training
- [ ] Documented feature meanings

---

**Remember:** More features ≠ better performance. Start with ~80-100 well-chosen features based on physics!
