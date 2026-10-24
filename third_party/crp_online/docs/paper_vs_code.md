# Paper vs code: where they differ and what we followed

Sources: paper = Durasevic, Dumic & Gil-Gala (2026), Table 3, + CEC paper; code = authors' C++
(crp, crpecf) + ECF 1.4.2. "Followed" = what this Python reproduction does.

| # | Topic | Paper | Code | Followed |
|---|---|---|---|---|
| 1 | GP initialisation | ramped half-and-half | 50% full / 50% grow, always at max depth, not ramped (Tree.cpp, "TODO: ramped") | code |
| 2 | Mutation operators | 6 ops (subtree, hoist, node complement, node replacement, permutation, shrink), no gauss, probability 0.3 | 7 ops incl. gauss, one chosen uniformly; gauss always fails without constants, so 1/7 of mutations change nothing (effective rate ~0.26) | code |
| 3 | Crossover operators | 5 ops (simple, context-preserving, uniform, size-fair, one-point) | same 5, one chosen uniformly (crx.* lines in params are outside `<Genotype>`, so ignored) | same in both |
| 4 | Max depth 5 | max depth 5, counting not specified | root = depth 0, so up to 6 levels / 63 nodes | code |
| 5 | Crossover failure | not stated | after 5 tries over the depth limit it gives up; child stays a copy of the worst | code |
| 6 | Tournament | steady-state tournament, size 3 | 3 random picks with repeats allowed; worst = first of the worst on ties | code |
| 7 | Training set | 840 instances, "same properties" as Caserta | generator's active settings: 3000 x 10x10, clock seed | paper: 21 classes x 40, seed 2026 (no yard identical to a test yard, checked) |
| 8 | Hidden IDs (Maximum) | Maximum = last visible + 1 | Maximum commented out, -1 (Ignore) active | paper (Maximum, switched in both places) |
| 9 | PU1/PU2 "one-container" check | not stated | `secondSmallest=!INT_MAX` assigns instead of compares, so it means "height == 1" | code (matches all 8 paper totals) |
| 10 | Subtraction order | not stated | `Execute() - Execute()`: C++ evaluation order unspecified | left - right |
| 11 | Ignore edge cases | not stated | AVG 0/0 and INT_MAX - (-1) overflow are undefined behaviour | not relevant: only Maximum is used, Ignore is not ported |
| 12 | Unrestricted at alpha = 0 | ~29,000 | preventive check can never fire (equal hidden values) | open issue, alpha = 0 not used |
| 13 | Budget | 50,000 evaluations | 50 generations after an initial population = 51,000 | code (51,000) |
| 14 | Grow initialisation | not stated | root always an operator (initmindepth 1) | code |
| 15 | One-point crossover | not stated | skips one extra node after a shape mismatch | code |
| 16 | Permutation mutation | not stated | with 2 children: swaps only 50% of the time | code |
| 17 | Subtree mutation | not stated | fails if a valid node is found only on the 4th try | code |
| 18 | Best rule kept | not stated | updated only at generation ends; last best on ties; only strictly better replaces | code |
