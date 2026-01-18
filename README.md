# Course-Optimizer
The system represents academic constraints—including prerequisites, corequisites, time conflicts, credit limits, electives, and degree requirements—through two distinct modeling approaches: a pandas-based relational structure and a NetworkX-based graph formulation. The integrated planning problem is subsequently solved using Google OR-Tools CP-SAT under three optimization paradigms: multi-weighted objective functions, lexicographic optimization, and Pareto frontier enumeration.


Parallel implementations of the academic planning model, consisting of a pandas DataFrame–driven tabular formulation and a NetworkX-based graph formulation, developed as separate files to study expressiveness, constraint encoding, and solver behavior under different representations.

