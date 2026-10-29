# Validation status

- Three automated tests pass using synthetic fixtures: original Matrix Market/metadata loading; mixed targets and early-only expression; ambiguous clone exclusion; fractional predictions and train-only gene filtering.
- Real author metadata was downloaded and its column names and cell labels checked.
- The source count file is 2,065,554,808 bytes compressed. A full biological evaluation was not completed for this delivery; no real-data accuracy or performance improvement is claimed.
- The chunked loader was verified on a fixture matching the original format. Resource use on the full dataset remains to be measured.
- Dependencies are compatibility ranges, not an exact reproducible lockfile. Each actual run records versions and SHA256 hashes.
- The model estimates observed clone composition conditional on two selected fates, not individual-cell fate, intervention effects, or clinical outcomes.
- Remote GitHub repository was not created: the connected account is AndriyPo, and available GitHub tools lack repository creation. README provides Fainahehe publishing steps.
