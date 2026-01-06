import pandas as pd
import networkx as nx 
from ortools.sat.python import cp_model 
from graph_builder import graph

courses_list = list(graph.g.nodes)
semesters_list = [1,2,3,4,5,6,7,8]

#building model
model = cp_model.CpModel()

#creating decision variables(each variable has a domain{0,1})
x = {}
for c in courses_list:
    for s in semesters_list:
        x[(c,s)] = model.NewBoolVar(f"x_{c}_{s}")

#Hard Constraint 1(each course must be taken atmost once)
for c in courses_list:
    model.Add(sum(x[(c,s)] for s in semesters_list) <= 1)

#Hard Constraint 2(required courses for a particular major must be taken exactly once)
majors_list = list(graph.g.graph["majors"]["required"].keys())
for i,value in enumerate(majors_list):
    print(f"{i+1}. {value}")
p = int(input("Your selected preference number: "))
major_pref = majors_list[p-1]

for c in graph.g.graph["majors"]["required"][major_pref]:
    model.Add(sum(x[c,s] for s in semesters_list)== 1)

#Hard Constraint 3(credit limits)
max_credits = int(input("Enter max credits:"))
min_credits = int(input("Enter min credits:"))
graph.AddCreditLimits(max_credits,min_credits)
semester_credits = {}
for s in semesters_list:
    semester_credits[s] = sum(x[(c,s)]*graph.g.nodes[c]["credits"] for c in courses_list)
    model.Add(semester_credits[s] <= max_credits)
    model.Add(semester_credits[s] >= min_credits)


#Hard Constraint 4(prerequisites)
for u,v,data in graph.g.edges(data = True):
    if data["type"] == "prerequisite":
        for s in semesters_list:
            model.Add(sum(x[(u,sp)] for sp in semesters_list if sp<s) >= x[(v,s)])

#Hard Constraint 5(Time conflicts)
for u,v,data in graph.g.edges(data = True):
    if data["type"] == "conflict" and u<v:
        for s in semesters_list:
            model.Add(x[(u,s)] + x[(v,s)] <= 1)

#Hard Constraint 6(Electives)
for s in range(5,9):
    model.Add(sum(x[(c,s)] for c in graph.g.graph["open_electives"]) <= 1)
for s in range(5,9):
    model.Add(sum(x[(c,s)] for c in graph.g.graph["majors"]["electives"][major_pref]) == 1)

#Hard Constraint 7(Co-requisites)
for u,v,data in graph.g.edges(data= True):
    if data["type"] == "corequisite":
        for s in semesters_list:
            model.Add(x[(u,s)]<= x[(v,s)])

#Hard Constraint 8(Semester Availabilty)
for c in courses_list:
    for s in semesters_list:
        if s not in graph.g.nodes[c]["availability"]:
            model.Add(x[(c,s)]== 0)

#1.Must Optimise objective(workload variance across all semesters):
workload = {}
for s in semesters_list:
    workload[s] = sum(x[(c, s)]*int(graph.g.nodes[c]["difficulty"]) for c in courses_list)
diff = {}
for i in semesters_list:
    for j in semesters_list:
        if i < j:
            d = model.NewIntVar(0, 50, f"diff_{i}_{j}")
            diff[(i,j)] = d
            model.Add(d >= workload[i] - workload[j])
            model.Add(d >= workload[j] - workload[i])
penalty_workload = 15*sum(diff.values())


#2a.Morning classes(before 10)
penalty_morn = sum(x[(c,s)]*10  for s in semesters_list
                                for c in courses_list 
                                for a in ("lecture","tutorial","practical")
                                for b in range(3)
                                if graph.g.nodes[c][a][b][2] != None and int(graph.g.nodes[c][a][b][2]) <= 10)

#2b.Classes after 5

penalty_eve = sum(x[(c,s)]*7  for s in semesters_list
                              for c in courses_list
                              for a in ("lecture","tutorial","practical")
                              for b in range(3)
                              if graph.g.nodes[c][a][b][1] != None and int(graph.g.nodes[c][a][b][1]) >= 17)

penalty_timings = penalty_morn + penalty_eve

#3.Minimize gaps
slots = []

for c in courses_list:
    for q in ("lecture","tutorial","practical"):
        for r in range(3):
            day, start, end = graph.g.nodes[c][q][r]
            if day is not None:
                slots.append((day, int(start), int(end), c))

from collections import defaultdict
day_slots = defaultdict(list)

for day, start, end, course in slots:
    day_slots[day].append((start, end, course))

gap_penalty_vars = []

for day, items in day_slots.items():
    items.sort(key=lambda x: x[0])
    for i in range(len(items) - 1):
        s1, e1, c1 = items[i]
        s2, e2, c2 = items[i + 1]
        gap = s2 - e1

        if gap > 0:
            for s in semesters_list:
                g = model.NewBoolVar(f"gap_{c1}_{c2}_{s}")
                model.Add(g <= x[(c1, s)])
                model.Add(g <= x[(c2, s)])
                model.Add(g >= x[(c1, s)] + x[(c2, s)] - 1)
                gap_penalty_vars.append(gap * g)

penalty_gaps = 5*sum(gap_penalty_vars)


#4.Fairness Imbalance
day_diff_vars = []

days = list(day_slots.keys())

for s in semesters_list:
    day_workload = {}
    for day in days:
        day_workload[day] = sum(x[(c, s)]*graph.g.nodes[c]["difficulty"] for (_, _, c) in day_slots[day])

    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            d = model.NewIntVar(0, 100, f"day_diff_{days[i]}_{days[j]}_s{s}")
            model.Add(d >= day_workload[days[i]] - day_workload[days[j]])
            model.Add(d >= day_workload[days[j]] - day_workload[days[i]])
            day_diff_vars.append(d)
    
imbalance = 3*sum(day_diff_vars)

model.Minimize(penalty_workload + penalty_timings + penalty_gaps + imbalance)
solver = cp_model.CpSolver()
solver.Solve(model)
#store baseline solution
x0 = {(c,s): solver.Value(x[(c,s)]) for c in courses_list for s in semesters_list}

#triggers for reoptimization
def perf_feas():
    if solver.Value(penalty_workload)>500 or solver.Value(penalty_timings)>500 or solver.Value(penalty_gaps)>500 or solver.Value(imbalance)>500:
        return True
    return False
def ext_feas():
    ans = input("Did any course become unavailable")
    if ans.lower() in {"YES","Yes","yes","yea","yeah"}:
        n = input("How many courses became unavailable?")
        affected_semesters = []
        for _ in range(n):
            c = input("Course name[CAPS]: ")
            s = input("Semester: ")
            affected_semesters.append(int(s))
            if s in graph.g.nodes[c]["availability"]:
                graph.g.nodes[c]["availability"].remove(s)
        global current_semester
        current_semester = min(affected_semesters)
        return True
    return False

if perf_feas() is True or ext_feas() is True:
    #model for reoptmization
    robust_model = cp_model.CpModel()

    #creating decision variables(each variable has a domain{0,1})
    x = {}
    for c in courses_list:
        for s in semesters_list:
            x[(c,s)] = robust_model.NewBoolVar(f"x_{c}_{s}")

    #Hard Constraint 1(each course must be taken atmost once)
    for c in courses_list:
        robust_model.Add(sum(x[(c,s)] for s in semesters_list) <= 1)

    #Hard Constraint 2(required courses for a particular major must be taken exactly once)
    majors_list = list(graph.g.graph["majors"]["required"].keys())
    major_pref = majors_list[p-1]

    for c in graph.g.graph["majors"]["required"][major_pref]:
        robust_model.Add(sum(x[c,s] for s in semesters_list)== 1)

    #Hard Constraint 3(credit limits)
    semester_credits = {}
    for s in semesters_list:
        semester_credits[s] = sum(x[(c,s)]*graph.g.nodes[c]["credits"] for c in courses_list)
    E = {}
    for s in semesters_list:
        E[s] = robust_model.NewIntVar(0,15,f"epsilon_{s}")
        robust_model.Add(semester_credits[s] <= max_credits + E[s])
        robust_model.Add(semester_credits[s] >= min_credits - E[s])
    epsilon = sum(E.values())*0.8

    #Hard Constraint 4(prerequisites)
    for u,v,data in graph.g.edges(data = True):
        if data["type"] == "prerequisite":
            for s in semesters_list:
                robust_model.Add(sum(x[(u,sp)] for sp in semesters_list if sp<s) >= x[(v,s)])

    #Hard Constraint 5(Time conflicts)
    for u,v,data in graph.g.edges(data = True):
        if data["type"] == "conflict" and u<v:
            for s in semesters_list:
                robust_model.Add(x[(u,s)] + x[(v,s)] <= 1)

    #Hard Constraint 6(Electives)
    for s in range(5,9):
        robust_model.Add(sum(x[(c,s)] for c in graph.g.graph["open_electives"]) <= 1)
    for s in range(5,9):
        robust_model.Add(sum(x[(c,s)] for c in graph.g.graph["majors"]["electives"][major_pref]) == 1)

    #Hard Constraint 7(Co-requisites)
    for u,v,data in graph.g.edges(data= True):
        if data["type"] == "corequisite":
            for s in semesters_list:
                robust_model.Add(x[(u,s)]<= x[(v,s)])

    #Hard Constraint 8(Semester Availabilty)
    for c in courses_list:
        for s in semesters_list:
            if s not in graph.g.nodes[c]["availability"]:
                robust_model.Add(x[(c,s)]== 0)


                        
    #1.Must Optimise objective(workload variance across all semesters):
    workload = {}
    for s in semesters_list:
        workload[s] = sum(x[(c, s)]*int(graph.g.nodes[c]["difficulty"]) for c in courses_list)
    diff = {}
    for i in semesters_list:
        for j in semesters_list:
            if i < j:
                d = robust_model.NewIntVar(0, 50, f"diff_{i}_{j}")
                diff[(i,j)] = d
                robust_model.Add(d >= workload[i] - workload[j])
                robust_model.Add(d >= workload[j] - workload[i])
    penalty_workload = 15*sum(diff.values())


    #2a.Morning classes(before 10)
    penalty_morn = sum(x[(c,s)]*10  for s in semesters_list
                                    for c in courses_list 
                                    for a in ("lecture","tutorial","practical")
                                    for b in range(3)
                                    if graph.g.nodes[c][a][b][2] != None and int(graph.g.nodes[c][a][b][2]) <= 10)

    #2b.Classes after 5

    penalty_eve = sum(x[(c,s)]*7 for s in semesters_list
                                 for c in courses_list
                                 for a in ("lecture","tutorial","practical")
                                 for b in range(3)
                                 if graph.g.nodes[c][a][b][1] != None and int(graph.g.nodes[c][a][b][1]) >= 17)

    penalty_timings = penalty_morn + penalty_eve

    #3.Minimize gaps
    slots = []

    for c in courses_list:
        for q in ("lecture","tutorial","practical"):
            for r in range(3):
                day, start, end = graph.g.nodes[c][q][r]
                if day is not None:
                    slots.append((day, int(start), int(end), c))

    from collections import defaultdict
    day_slots = defaultdict(list)

    for day, start, end, course in slots:
        day_slots[day].append((start, end, course))

    gap_penalty_vars = []

    for day, items in day_slots.items():
        items.sort(key=lambda x: x[0])
        for i in range(len(items) - 1):
            s1, e1, c1 = items[i]
            s2, e2, c2 = items[i + 1]
            gap = s2 - e1

            if gap > 0:
                for s in semesters_list:
                    g = robust_model.NewBoolVar(f"gap_{c1}_{c2}_{s}")
                    robust_model.Add(g <= x[(c1, s)])
                    robust_model.Add(g <= x[(c2, s)])
                    robust_model.Add(g >= x[(c1, s)] + x[(c2, s)] - 1)
                    gap_penalty_vars.append(gap * g)

    penalty_gaps = 5*sum(gap_penalty_vars)


    #4.Fairness Imbalance
    day_diff_vars = []

    days = list(day_slots.keys())

    for s in semesters_list:
        day_workload = {}
        for day in days:
            day_workload[day] = sum(x[(c, s)]*graph.g.nodes[c]["difficulty"] for (_, _, c) in day_slots[day])

        for i in range(len(days)):
            for j in range(i + 1, len(days)):
                d = robust_model.NewIntVar(0, 100, f"day_diff_{days[i]}_{days[j]}_s{s}")
                robust_model.Add(d >= day_workload[days[i]] - day_workload[days[j]])
                robust_model.Add(d >= day_workload[days[j]] - day_workload[days[i]])
                day_diff_vars.append(d)
        
            
    imbalance = 3*sum(day_diff_vars)

    #Robust model requirements
    for c in courses_list:
        for s in semesters_list:
            if s < current_semester:
                robust_model.Add(x[(c, s)] == x0[(c,s)])

    delta = {}

    for c in courses_list:
        for s in semesters_list:
            delta[(c,s)] = robust_model.NewIntVar(0,6,f"delta_{c}_{s}")
            robust_model.Add(delta[(c,s)] >= x[(c,s)] - x0[(c,s)])
            robust_model.Add(delta[(c,s)] >= x0[(c,s)] - x[(c,s)])
    penalty_stability = sum((9 - s)*delta[(c, s)]for c in courses_list for s in semesters_list)

    robust_model.Minimize(penalty_workload + penalty_timings + penalty_gaps + imbalance 
                          + penalty_stability 
                          + epsilon)
    
    #Pareto Solutions
    class ParetoCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self, objectives, x_vars):
            super().__init__()
            self.objectives = objectives
            self.x_vars = x_vars
            self.solutions = []

        def OnSolutionCallback(self):
            obj_vals = tuple(self.Value(o) for o in self.objectives)
            assignment = {
                (c, s): 1
                for (c, s), v in self.x_vars.items()
                if self.Value(v) == 1
            }
            self.solutions.append((obj_vals, assignment))

    robust_objectives = [penalty_workload,penalty_timings,penalty_gaps,imbalance,penalty_stability,epsilon]
    robust_callback = ParetoCallback(robust_objectives,x)
    robust_solver = cp_model.CpSolver()
    robust_solver.parameters.enumerate_all_solutions = True
    robust_solver.parameters.max_time_in_seconds = 15
    
    solver.Solve(robust_model, robust_callback)

    def is_dominated(sol, others):
        return any(
            all(o <= s for o, s in zip(other[0], sol[0])) and
            any(o < s for o, s in zip(other[0], sol[0]))
            for other in others
        )

    pareto_solutions = [
        sol for sol in robust_callback.solutions
        if not is_dominated(sol, robust_callback.solutions)
    ]

    for i, (obj, assign) in enumerate(pareto_solutions, 1):
        print(f"Solution {i}")
        print(f"workload={obj[0]} timing={obj[1]} gaps={obj[2]} imbalance={obj[3]}")
        for s in semesters_list:
            courses = [c for (c, sem) in assign if sem == s]
            if courses:
                print(f"Semester {s}: {courses}")
        print("*" * 40)

else:
    #Pareto Solutions
    class ParetoCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self, objectives, x_vars):
            super().__init__()
            self.objectives = objectives
            self.x_vars = x_vars
            self.solutions = []

        def OnSolutionCallback(self):
            obj_vals = tuple(self.Value(o) for o in self.objectives)
            assignment = {
                (c, s): 1
                for (c, s), v in self.x_vars.items()
                if self.Value(v) == 1
            }
            self.solutions.append((obj_vals, assignment))
    baseline_objectives = [penalty_workload,penalty_timings,penalty_gaps,imbalance]
    baseline_callback = ParetoCallback(baseline_objectives,x)
    baseline_solver = cp_model.CpSolver()
    baseline_solver.parameters.enumerate_all_solutions = True
    baseline_solver.parameters.max_time_in_seconds = 15
    solver.Solve(model, baseline_callback)

    def is_dominated(sol, others):
        return any(
            all(o <= s for o, s in zip(other[0], sol[0])) and
            any(o < s for o, s in zip(other[0], sol[0]))
            for other in others
        )

    pareto_solutions = [
        sol for sol in baseline_callback.solutions
        if not is_dominated(sol, baseline_callback.solutions)
    ]

    for i, (obj, assign) in enumerate(pareto_solutions, 1):
        print(f"Solution {i}")
        print(f"workload={obj[0]} timing={obj[1]} gaps={obj[2]} imbalance={obj[3]}")
        for s in semesters_list:
            courses = [c for (c, sem) in assign if sem == s]
            if courses:
                print(f"Semester {s}: {courses}")
        print("*" * 40)