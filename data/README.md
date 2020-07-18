# Descriptors Calculation

## Solvent Descriptors

The descriptors are from the paper "[Machine learning and molecular descriptors enable rational solvent selection in asymmetric catalysis](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c9sc01844a#!divAbstract)" by Amar et al.

## Ligand and Base Descriptors

Descriptors are calculated using [COSMOquick](https://www.3ds.com/products-services/biovia/products/molecular-modeling-simulation/solvation-chemistry/cosmoquick/).  The QSPR & ADME option is used to calculate the sigma moment descriptors which are named as follows:

- 'area' for the zero sigma moment
- 'M2' for the second sigma moment
- 'M3' for the third sigma moment
- 'Macc3' for the hydrogen bond acceptor strength
- 'Mdon3' for the hydrogen bond donor strength

Additionally the solubility in 2 Me-THF is predicted.  All calculations are done at 25°C

