import pandas as pd
import networkx as nx 
from ortools.sat.python import cp_model 
from  graph_builder import graph
import time

courses_list = list(graph.g.nodes)
semesters_list = [1,2,3,4,5,6,7,8]
current_semester = min(semesters_list)

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
min_courses_per_sem = 3
max_courses_per_sem = 6  

for s in semesters_list:
    model.Add(sum(x[(c,s)] for c in courses_list) >= min_courses_per_sem)
    model.Add(sum(x[(c,s)] for c in courses_list) <= max_courses_per_sem)

import random
extra_semesters = random.sample(semesters_list, 2)
for s in extra_semesters:
    model.Add(sum(x[(c, s)] for c in courses_list) >= 4)

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
    model.Add(sum(x[(c,s)] for c in graph.g.graph["majors"]["electives"][major_pref]) <= 1)

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
avg = model.NewIntVar(0, 1000, "avg_workload")
model.Add(avg * 8 >= sum(workload.values()) - 50)
model.Add(avg * 8 <= sum(workload.values()) + 50)

devs = []
for s in semesters_list:
    d = model.NewIntVar(0, 1000, f"dev_{s}")
    model.Add(d >= workload[s] - avg)
    model.Add(d >= avg - workload[s])
    devs.append(d)

penalty_workload = model.NewIntVar(0, 10_000_000, "penalty_workload")
model.Add(penalty_workload == sum(devs))

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

penalty_timings = model.NewIntVar(0, 10_000_000, "penalty_timings")
model.Add(penalty_timings == penalty_morn + penalty_eve)

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

penalty_gaps = model.NewIntVar(0, 10_000_000, "penalty_gaps")
model.Add(penalty_gaps == sum(gap_penalty_vars))


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
    
imbalance = model.NewIntVar(0, 10_000_000, "imbalance")
model.Add(imbalance == sum(day_diff_vars))

model.Minimize(penalty_workload + penalty_timings + penalty_gaps + imbalance)
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30
solver.Solve(model)
#store baseline solution
x0 = {(c,s): solver.Value(x[(c,s)]) for c in courses_list for s in semesters_list}

#triggers for reoptimization
def perf_feas():
    if solver.Value(penalty_workload)>500 or solver.Value(penalty_timings)>500 or solver.Value(penalty_gaps)>500 or solver.Value(imbalance)>500:
        return True
    return False
def ext_feas():
    ans = input("Did any course become unavailable? : ")
    if ans in ("YES","Yes","yes","yea","yeah"):
        n = int(input("How many courses became unavailable? : "))
        affected_semesters = []
        for _ in range(n):
            c = input("Course name[CAPS]: ")
            s = int(input("Semester: "))
            affected_semesters.append(int(s))
            if s in graph.g.nodes[c]["availability"]:
                graph.g.nodes[c]["availability"].remove(s)
        global current_semester
        current_semester = min(affected_semesters)
        return True
    return False

perf = perf_feas()
ext = ext_feas()

if perf or ext:
    print("Reoptimization triggered:", perf or ext)
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
    min_courses_per_sem = 3
    max_courses_per_sem = 6 

    for s in semesters_list:
        robust_model.Add(sum(x[(c,s)] for c in courses_list) >= min_courses_per_sem)
        robust_model.Add(sum(x[(c,s)] for c in courses_list) <= max_courses_per_sem)
    
    import random
    extra_semesters = random.sample(semesters_list, 2)
    for s in extra_semesters:
        model.Add(sum(x[(c, s)] for c in courses_list) >= 4)

    #Hard Constraint 2(required courses for a particular major must be taken exactly once)
    majors_list = list(graph.g.graph["majors"]["required"].keys())
    major_pref = majors_list[p-1]

    for c in graph.g.graph["majors"]["required"][major_pref]:
        robust_model.Add(sum(x[c,s] for s in semesters_list)== 1)

    #Hard Constraint 3(credit limits)
    semester_credits = {}
    for s in semesters_list:
        semester_credits[s] = sum(x[(c,s)]*graph.g.nodes[c]["credits"] for c in courses_list)
    
    for s in semesters_list:
        robust_model.Add(semester_credits[s] <= max_credits)
        robust_model.Add(semester_credits[s] >= min_credits)
    

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
        robust_model.Add(sum(x[(c,s)] for c in graph.g.graph["majors"]["electives"][major_pref]) <= 1)

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
    avg = robust_model.NewIntVar(0, 1000, "avg_workload")
    robust_model.Add(avg * len(semesters_list) >= sum(workload.values()) - 50)
    robust_model.Add(avg * len(semesters_list) <= sum(workload.values()) + 50)

    devs = []
    for s in semesters_list:
        d = robust_model.NewIntVar(0, 1000, f"dev_{s}")
        robust_model.Add(d >= workload[s] - avg)
        robust_model.Add(d >= avg - workload[s])
        devs.append(d)

    penalty_workload = robust_model.NewIntVar(0, 10_000_000, "penalty_workload")
    robust_model.Add(penalty_workload == sum(devs))


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

    penalty_timings = robust_model.NewIntVar(0, 10_000_000, "penalty_timings")
    robust_model.Add(penalty_timings == penalty_morn + penalty_eve)

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

    penalty_gaps = robust_model.NewIntVar(0, 10_000_000, "penalty_gaps")
    robust_model.Add(penalty_gaps ==  sum(gap_penalty_vars))


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
        
            
    imbalance = robust_model.NewIntVar(0, 10_000_000, "imbalance")
    robust_model.Add(imbalance == sum(day_diff_vars))

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
    penalty_stability = robust_model.NewIntVar(0, 10_000_000, "penalty_stability")
    robust_model.Add(penalty_stability == sum((9 - s)*delta[(c, s)] for c in courses_list for s in semesters_list))

    model.Minimize(
        800*penalty_workload +
        500*penalty_timings +
        200*penalty_gaps +
        100*imbalance +
        500*penalty_stability  
    )
    robust_solver = cp_model.CpSolver()
    robust_solver.parameters.max_time_in_seconds = 5
    robust_solver.parameters.num_search_workers = 8

    status = robust_solver.Solve(robust_model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Robust model infeasible")
        exit()

    print("Robust Solution")
    print(
        f"workload={robust_solver.Value(penalty_workload)} "
        f"timing={robust_solver.Value(penalty_timings)} "
        f"gaps={robust_solver.Value(penalty_gaps)} "
        f"imbalance={robust_solver.Value(imbalance)}"
    )

    for s in semesters_list:
        courses = [
            c for (c, sem), v in x.items()
            if sem == s and robust_solver.Value(v) == 1
        ]
        if courses:
            print(f"Semester {s}: {courses}")

else:
    model.Minimize(
        800*penalty_workload +
        500*penalty_timings +
        200*penalty_gaps +
        100*imbalance   
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Baseline model infeasible")
        exit()

    print("Baseline Solution")
    print(
        f"workload={solver.Value(penalty_workload)} "
        f"timings={solver.Value(penalty_timings)} "
        f"gaps={solver.Value(penalty_gaps)} "
        f"imbalance={solver.Value(imbalance)}"
    )

    for s in semesters_list:
        courses = [
            c for (c, sem), v in x.items()
            if sem == s and solver.Value(v) == 1]
        if courses:
            print(f"Semester {s}: {courses}")