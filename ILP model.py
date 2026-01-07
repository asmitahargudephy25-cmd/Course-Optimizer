import pandas as pd
df = pd.read_csv("course_catalog.csv")
courses_list = df["course_name"].to_list()
semesters_list = [1,2,3,4,5,6,7,8]
difficulty = df.set_index("course_name")["difficulty"].to_dict()
current_semester = min(semesters_list)
credits_dict = df.set_index("course_name")["credits"].to_dict()

max_credits = int(input("Enter max credits:"))
min_credits = int(input("Enter min credits:"))

from ortools.sat.python import cp_model 

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
df2 = pd.read_csv("major_requirements.csv")
majors_list = df2["major"].to_list()
for index,value in enumerate(majors_list):
    print(index +1,value)
q = int(input("your selected option is:"))
major_pref = majors_list[q-1]

req_string = str(df2[df2["major"] == major_pref]["required_courses"].values[0])
for c in req_string.split("_") :
    model.Add(sum(x[(c,s)] for s in semesters_list) == 1)

#Hard Constraint 3(credit limits)
semester_credits = {}
for s in semesters_list:
    semester_credits[s] = sum(x[(c,s)] * int(credits_dict[c]) for c in courses_list)
    model.Add(semester_credits[s] <= max_credits)
    model.Add(semester_credits[s] >= min_credits)


#Hard Constraint 4(prerequisites)
new_df = df[df["prerequisites"] != "NONE"]
prereq_dict = new_df.set_index("course_name")["prerequisites"].to_dict()
for k in prereq_dict:
    prereq_dict[k] = prereq_dict[k].split("_")

for key in prereq_dict.keys():
    for p in prereq_dict[key]:
        for s in semesters_list:
            model.Add(x[key,s]<= sum(x[p,t] for t in range(1,s)))

#Hard Constraint 5(Time conflicts)
from itertools import product

def normalize_and_pad(slot_str):
    if slot_str == "NONE":
        slots = []
    else:
        slots = [s.split("_") for s in slot_str.split("/")]

    slots = [tuple(p + [None] * (3 - len(p))) for p in slots]

    while len(slots) < 3:
        slots.append((None, None, None))

    return tuple(slots[:3])


y = {}

for c in courses_list:
    row = df[df["course_name"] == c].iloc[0]
    y[c] = {
        "l": normalize_and_pad(row["lecture"]),
        "t": normalize_and_pad(row["tutorial"]),
        "p": normalize_and_pad(row["practical"])
    }

LTP = ("l", "t", "p")

for a, b in product(courses_list, repeat=2):
    if a >= b:
        continue

    for k1, k2 in product(LTP, repeat=2):
        for i in range(3):
            for j in range(3):

                day_a, start_a, end_a = y[a][k1][i]
                day_b, start_b, end_b = y[b][k2][j]

                if day_a is None or day_b is None:
                    continue

                if day_a == day_b and not (end_a <= start_b or end_b <= start_a):
                    for s in semesters_list:
                        model.Add(x[(a, s)] + x[(b, s)] <= 1)

#Hard Constraint 6(Electives)
eldf = pd.read_csv("electives_catalog.csv")

for s in range(5,9):
    model.Add(sum(x[(c,s)] for c in eldf["open_electives"].to_list()) == 1)
for s in range(5,9):
    model.Add(sum(x[(c,s)] for c in eldf[major_pref].to_list()) == 1)

#Hard Constraint 7(Co-requisites)
for c in courses_list:
    for s in semesters_list:
            must = df[df["course_name"] == c]["corequisites"].values[0].split("|")
            for m in must:
                if m == "NONE":
                    continue
                else:
                    model.Add(x[(m,s)] >= x[(c,s)])

#Hard Constraint 8(Semester Availabilty)
sem_aval_list = df["semesters_available"].to_list()
for i, each in enumerate(sem_aval_list):
    sem_aval_list[i] = [int(s) for s in str(each).split("|")]
sem_aval = dict(zip(courses_list,sem_aval_list))
for c in courses_list:
    for s in semesters_list:
        if s not in sem_aval[c]:
            model.Add(x[(c,s)] == 0)
                    

#1.Must Optimise objective(workload variance across all semesters):
workload = {}
for s in semesters_list:
    workload[s] = sum(x[(c, s)]*difficulty[c] for c in courses_list)
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
penalty_morn = sum(x[(c, s)]*10 for s in semesters_list
                                for c in courses_list 
                                for a in LTP 
                                for b in range(3)
                                if y[c][a][b][2] != None and int(y[c][a][b][2]) <= 10)

#2b.Classes after 5

penalty_eve = sum(x[(c, s)]*7 for s in semesters_list
                              for c in courses_list
                              for a in LTP
                              for b in range(3)
                              if y[c][a][b][1] != None and int(y[c][a][b][1]) >= 17)

penalty_timimgs = penalty_morn + penalty_eve

#3.Minimize gaps
slots = []

for c in courses_list:
    for q in LTP:
        for r in range(3):
            day, start, end = y[c][q][r]
            if day is not None:
                slots.append((day, start, end, c))

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
        gap = int(s2) - int(e1)

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
        day_workload[day] = sum(x[(c, s)]*difficulty[c] for (_, _, c) in day_slots[day])

    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            d = model.NewIntVar(0, 100, f"day_diff_{days[i]}_{days[j]}_s{s}")
            model.Add(d >= day_workload[days[i]] - day_workload[days[j]])
            model.Add(d >= day_workload[days[j]] - day_workload[days[i]])
            day_diff_vars.append(d)
    
imbalance = 3*sum(day_diff_vars)

model.Minimize(penalty_workload + penalty_timimgs + penalty_gaps + imbalance)
solver = cp_model.CpSolver()
solver.Solve(model)
#store baseline solution
x0 = {(c,s): solver.Value(x[(c,s)]) for c in courses_list for s in semesters_list}

#triggers for reoptimization
def perf_feas():
    if solver.Value(penalty_workload)>500 or solver.Value(penalty_timimgs)>500 or solver.Value(penalty_gaps)>500 or solver.Value(imbalance)>500:
        return True
    return False
def ext_feas():
    ans = input("Did any course become unavailable")
    if ans in ("YES","Yes","yes","yea","yeah"):
        n = int(input("How many courses became unavailable?"))
        affected_semesters = []
        for _ in range(n):
            c = input("Course name[CAPS]: ")
            s = int(input("Semester: "))
            affected_semesters.append(s)
            if s in sem_aval[c]:
                sem_aval[c].remove(s)         
        global current_semester
        current_semester = min(affected_semesters)
        return True
    return False

p = perf_feas()

e = ext_feas()

if p or e:
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
    req_string = str(df2[df2["major"] == major_pref]["required_courses"].values[0])
    for c in req_string.split("_") :
        robust_model.Add(sum(x[(c,s)] for s in semesters_list) == 1)

    #Hard Constraint 3(credit limits)
    semester_credits = {}
    for s in semesters_list:
        semester_credits[s] = sum(x[(c,s)] * int(credits_dict[c]) for c in courses_list)
    E = {}
    for s in semesters_list:
        E[s] = robust_model.NewIntVar(0,15,f"epsilon_{s}")
        robust_model.Add(semester_credits[s] <= max_credits + E[s])
        robust_model.Add(semester_credits[s] >= min_credits - E[s])
    epsilon = sum(E.values())

    #Hard Constraint 4(prerequisites)
    new_df = df[df["prerequisites"] != "NONE"]
    prereq_dict = new_df.set_index("course_name")["prerequisites"].to_dict()
    for k in prereq_dict:
        prereq_dict[k] = prereq_dict[k].split("_")

    for key in prereq_dict.keys():
        for p in prereq_dict[key]:
            for s in semesters_list:
                robust_model.Add(x[key,s]<= sum(x[p,t] for t in range(1,s)))

    #Hard Constraint 5(Time conflicts)
    from itertools import product

    def normalize_and_pad(slot_str):
        if slot_str == "NONE":
            slots = []
        else:
            slots = [s.split("_") for s in slot_str.split("/")]

        slots = [tuple(p + [None] * (3 - len(p))) for p in slots]

        while len(slots) < 3:
            slots.append((None, None, None))

        return tuple(slots[:3])


    y = {}

    for c in courses_list:
        row = df[df["course_name"] == c].iloc[0]
        y[c] = {
            "l": normalize_and_pad(row["lecture"]),
            "t": normalize_and_pad(row["tutorial"]),
            "p": normalize_and_pad(row["practical"])
        }

    LTP = ("l", "t", "p")

    for a, b in product(courses_list, repeat=2):
        if a >= b:
            continue

        for k1, k2 in product(LTP, repeat=2):
            for i in range(3):
                for j in range(3):

                    day_a, start_a, end_a = y[a][k1][i]
                    day_b, start_b, end_b = y[b][k2][j]

                    if day_a is None or day_b is None:
                        continue

                    if day_a == day_b and not (end_a <= start_b or end_b <= start_a):
                        for s in semesters_list:
                            robust_model.Add(x[(a, s)] + x[(b, s)] <= 1)

    #Hard Constraint 5(Electives)
    eldf = pd.read_csv("electives_catalog.csv")

    for s in range(5,9):
        robust_model.Add(sum(x[(c,s)] for c in eldf["open_electives"].to_list()) == 1)
    for s in range(5,9):
        robust_model.Add(sum(x[(c,s)] for c in eldf[major_pref].to_list()) == 1)

    #Hard Constraint 6(Co-requisites)
    for c in courses_list:
        for s in semesters_list:
                must = df[df["course_name"] == c]["corequisites"].values[0].split("|")
                for m in must:
                    if m == "NONE":
                        continue
                    else:
                        robust_model.Add(x[(m,s)] >= x[(c,s)])

    #Hard Constraint 7(Semester Availabilty)
    sem_aval_list = df["semesters_available"].to_list()
    for i, each in enumerate(sem_aval_list):
        sem_aval_list[i] = [int(s) for s in str(each).split("|")]
    sem_aval = dict(zip(courses_list,sem_aval_list))
    for c in courses_list:
        for s in semesters_list:
            if s not in sem_aval[c]:
                robust_model.Add(x[(c,s)] == 0)


                        

    #1.Must Optimise objective(workload variance across all semesters):
    workload = {}
    for s in semesters_list:
        workload[s] = sum(x[(c, s)]*difficulty[c] for c in courses_list)
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
    penalty_morn = sum(x[(c, s)]*10 for s in semesters_list
                                    for c in courses_list 
                                    for a in LTP 
                                    for b in range(3)
                                    if y[c][a][b][2] != None and int(y[c][a][b][2]) <= 10)

    #2b.Classes after 5

    penalty_eve = sum(x[(c, s)]*7 for s in semesters_list
                                 for c in courses_list
                                 for a in LTP
                                 for b in range(3)
                                 if y[c][a][b][1] != None and int(y[c][a][b][1]) >= 17)

    penalty_timimgs = penalty_morn + penalty_eve

    #3.Minimize gaps
    slots = []

    for c in courses_list:
        for q in LTP:
            for r in range(3):
                day, start, end = y[c][q][r]
                if day is not None:
                    slots.append((day, start, end, c))

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
            gap = int(s2) - int(e1)

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
            day_workload[day] = sum(x[(c, s)]*difficulty[c] for (_, _, c) in day_slots[day])

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
            if int(s) < int(current_semester):
                robust_model.Add(x[(c, s)] == x0[(c,s)])


    delta = {}

    for c in courses_list:
        for s in semesters_list:
            delta[(c,s)] = robust_model.NewIntVar(0,6,f"delta_{c}_{s}")
            robust_model.Add(delta[(c,s)] >= x[(c,s)] - x0[(c,s)])
            robust_model.Add(delta[(c,s)] >= x0[(c,s)] - x[(c,s)])
    penalty_stability = sum((9 - s)*delta[(c, s)]for c in courses_list for s in semesters_list)

    robust_model.Minimize(penalty_workload + penalty_timimgs + penalty_gaps + imbalance 
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

    robust_objectives = [penalty_workload,penalty_timimgs,penalty_gaps,imbalance,penalty_stability,epsilon]
    robust_callback = ParetoCallback(robust_objectives,x)
    robust_solver = cp_model.CpSolver()
    robust_solver.parameters.enumerate_all_solutions = True
    robust_solver.parameters.max_time_in_seconds = 15
    
    robust_solver.Solve(robust_model, robust_callback)

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
    baseline_objectives = [penalty_workload,penalty_timimgs,penalty_gaps,imbalance]
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



