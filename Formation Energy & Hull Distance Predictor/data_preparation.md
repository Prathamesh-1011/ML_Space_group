# 📄 Data Preparation Methodology: Model-2

## 1. Data Source & Scope

The raw data was retrieved from the **Materials Project (MP)** database via the `mp-api` (v2024+ standard). The scope was limited to all available inorganic compounds to ensure the widest possible chemical diversity for the deep learning model.

## 2. Feature Selection (Input/Output)

To align with the requirements for transformer-based and graph-based attention models, we strictly extracted:

* **Primary Input:** Reduced chemical formulas (compositional data only).
* **Target 1:** Formation energy per atom ().
* **Target 2:** Energy above hull ().
* **Target 3 (Classification):** Thermodynamic stability (Binary label).

## 3. Critical Preprocessing & Deduplication

A major challenge in materials data is **polymorphism** (multiple structures for the same chemical formula). Because Model-2 is composition-only, multiple entries for one formula would create "label noise."

* **Lowest-Energy Selection:** Following the Roost/CrabNet protocol, we sorted all entries by their energy above the hull.
* **Deduplication:** For every unique composition, only the **ground-state entry** (the polymorph with the lowest energy above hull) was retained. This ensures the model learns the most stable representation of a chemical system.

## 4. Data Cleaning & Validation

* **Null Removal:** Any entries missing formation energy or hull energy values were purged.
* **Composition Parsing:** Formulas were validated using the `pymatgen` library to ensure they could be parsed into element-fraction dictionaries.
* **Stability Thresholding:** A stability label (`is_stable`) was derived using a threshold of . Entries below this threshold are categorized as experimentally reachable or stable.

## 5. Final Dataset Structure

The resulting CSV is a flat file designed for rapid loading into PyTorch or TensorFlow dataloaders.

| Column | Description | Role |
| --- | --- | --- |
| `material_id` | Unique MP identifier | Traceability |
| `formula` | Pretty-printed chemical formula | Human-readable |
| `composition` | Reduced formula (e.g., ) | **Primary Input** |
| `formation_energy_per_atom` | Energy required to form the phase | Regression Target |
| `energy_above_hull` | Distance from thermodynamic equilibrium | Regression Target |
| `is_stable` | Binary classification (1 = Stable, 0 = Unstable) | Classification Target |
| `elements` | List of constituent elements | Metadata |
| `fractions` | Normalized atomic fractions | Metadata |

## 6. Summary Statistics

* **Raw Records:** 210579
* **Final Unique Compositions:** 150,202
* **Chemical Space:** Covers the majority of the periodic table (excluding highly unstable transuranic elements).
