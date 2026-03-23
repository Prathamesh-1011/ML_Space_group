# Feature Importance Analysis for Space Group Prediction

![Top-20 Composition Descriptors](Top-20_composition_descriptors.png)

This figure presents the **top 20 composition-based descriptors contributing to the prediction of crystal space groups** using the Random Forest ensemble model.

The feature importance values are computed by averaging the importance scores from the two trained Random Forest models used in the ensemble. These importance scores quantify how much each descriptor contributes to reducing classification uncertainty during tree splits.

To improve interpretability, the importance values are **normalized to a scale of 0–100**, where the most influential feature receives a value of 100 and all other features are scaled relative to it.

## Interpretation of Key Features 

Several categories of descriptors dominate the importance ranking:

**1. Periodic Table Position Indicators**

Features such as:

- `MagpieData mean Column`
- `MagpieData mean MendeleevNumber`

capture an element’s position in the periodic table. These descriptors implicitly encode periodic trends that influence bonding behavior and crystal symmetry.

**2. Valence Electron Descriptors**

Descriptors such as:

- `MagpieData mean NpValence`
- `MagpieData avg_dev NpValence`
- `MagpieData mean NdValence`

reflect the distribution of valence electrons within a compound. Valence electron configurations strongly influence the types of bonds formed between atoms and therefore affect the symmetry constraints of the resulting crystal structure.

**3. Electronegativity Statistics**

Features including:

- `MagpieData mean Electronegativity`
- `MagpieData avg_dev Electronegativity`

capture the degree of ionic versus covalent character in chemical bonds. Materials with large electronegativity differences tend to form more directional bonding patterns, which can influence crystal symmetry and preferred lattice arrangements.

**4. Structural and Packing Proxies**

Descriptors such as:

- `volume_per_atom`
- `packing_proxy`
- `MagpieData avg_dev GSvolume_pa`

provide indirect information about atomic packing efficiency and geometric constraints within the crystal lattice.

**5. Atomic Size and Bonding Descriptors**

Properties such as:

- `MagpieData mean CovalentRadius`
- `MagpieData avg_dev CovalentRadius`

reflect atomic size mismatches between elements in a compound, which can influence coordination environments and lattice distortions.

## Significance

The dominance of periodic table position, valence electron counts, and electronegativity descriptors indicates that the model primarily relies on **fundamental chemical principles governing bonding and atomic interactions** when predicting space group symmetry.

These results demonstrate that even without explicit structural information, composition-based features can capture meaningful signals related to crystal symmetry formation.

This observation aligns with prior work in materials informatics where composition-driven models such as **Magpie-based feature systems** have shown strong predictive capability for crystallographic properties. :contentReference[oaicite:0]{index=0}
