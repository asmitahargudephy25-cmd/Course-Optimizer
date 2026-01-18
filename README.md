# Course-Optimizer
OVERVIEW-

This project implements a constraint-based academic course planning system designed to generate valid, optimized multi-semester study plans under complex academic rules modeling real-world university constraints. The system represents academic constraints—including prerequisites, corequisites, time conflicts, credit limits, electives, and degree requirements—through two distinct modeling approaches: a pandas-based relational structure and a NetworkX-based graph formulation. The integrated planning problem is subsequently solved using Google OR-Tools CP-SAT under three optimization paradigms: multi-weighted objective functions, lexicographic optimization, and Pareto frontier enumeration.


KEY CONTRIBUTIONS-

1. Two independent modeling approaches: Pandas DataFrame–based tabular formulation and NetworkX graph-based formulation

2. Constraint Programming (CP-SAT) solution using Google OR-Tools

3. Constraints Modeled:
Hard Constraints-

#Each course taken at most once
#Semester-wise minimum and maximum course limits
#Credit bounds per semester
#Prerequisite ordering across semesters
#Corequisite co-enrollment
#Time conflict avoidance
#Semester availability enforcement
#Major-specific required courses
#Elective limits per semester

Soft Constraints (Optimized)-

#Workload variance across semesters
#Penalties for: Early morning classes
                Late evening classes
                Gaps between classes in a day
                Weekly day-wise workload imbalance
                Schedule irregularity across semesters
                Stability penalty during re-optimization

4. Three optimization strategies:
        a.Multi-weighted objective optimization - All soft constraints are combined into a single weighted objective, allowing trade-offs between competing preferences.
        b.Lexicographic optimization - Objectives are optimized in a strict priority order, ensuring higher-priority goals are satisfied before considering others.
        c.Pareto-front enumeration - The system enumerates Pareto-optimal solutions, exposing trade-offs between conflicting objectives.

| File         | Modeling Strategy          | Best For              |
| ------------ | -------- | --------------- | --------------------- |
| CP_SAT_1.py  | Pandas   | Multi-weighted  | Balanced schedules    |
| CP_SAT_2.py  | Pandas   | Lexicographic   | Strict priorities     |
| CP_SAT_3.py  | Pandas   | Pareto frontier | Trade-off analysis    |
| CP_SAT_G1.py | NetworkX | Multi-weighted  | Complex relationships |
| CP_SAT_G2.py | NetworkX | Lexicographic   | Graph prereqs         |
| CP_SAT_G3.py | NetworkX | Pareto frontier | Research comparison   |

5. Dynamic Reoptimization when : 
A course becomes unavailable in a certain semester or the current solution exceeds predefined perfomance thresholds.
Previously fixed decisions (past semesters) are locked, while future semesters are re-optimized with stability penalties to minimize disruption.

6. CSV-driven design (no hardcoded courses, majors, or constraints)

QUICK START-

Python 3.8+
Google OR-Tools (pip install ortools)
Pandas, NetworkX (pip install pandas networkx)

ARCHITECTURE-

__pycache__
.venv
.vscode

CP_SAT/
|──CP_SAT_1.py(Multiweighted objective function)
|──CP_SAT_2.py(Lexicographic optimization)
|──CP_SAT_3.py(Pareto Optimization)

CP_SAT_G/
|──CP_SAT_G1.py(Multiweighted objective function)
|──CP_SAT_G2.py(Lexicographic optimization)
|──CP_SAT_G3.py(Pareto Optimization)
|──graph_builder(data sorting)

data/
|──course_catalog.csv(courses,credits,prerequisites,corequisites,sem availabilities)
|──electives_catalog.csv(department and open electives)
|──major_requirements.csv(required courses)

license
Readme.md
requirements.txt

PROJECT MOTIVATION -

This project was developed to explore how real-world academic planning problems
can be formulated as constraint satisfaction and optimization problems.
As a first-year student, the goal was to understand how abstract constraints
(prerequisites, credits, scheduling conflicts) translate into formal models
solved by industrial solvers.

SCOPE AND LIMITATIONS-

This project is a research and learning prototype and does not aim to:
- Replace official university registration systems
- Model all institution-specific academic policies
- Guarantee optimality under all possible objective weightings
The focus is on modeling, optimization strategies, and comparative formulations.

FUTURE EXTENSIONS-

1. Tailor it to particular domain preferences
2. Suggest effective tradeoff solutions for various employment fields
3. Integration with University API's
These extensions are proposed directions for future exploration
and are not currently implemented.

CONTACT-
Author : Asmita Hargude
email : asmita.hargude.phy25@itbhu.ac.in
linkedin : 