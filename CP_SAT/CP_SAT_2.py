import pandas as pd
import time
df = pd.read_csv("data/course_catalog.csv")
courses_list = df["course_name"].to_list()
semesters_list = [1,2,3,4,5,6,7,8]
difficulty = df.set_index("course_name")["difficulty"].astype(int).to_dict()
current_semester = min(semesters_list)
credits_dict = df.set_index("course_name")["credits"].to_dict()

max_credits = int(input("Enter max credits:"))
min_credits = int(input("Enter min credits:"))

from ortools.sat.python import cp_model 
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
time_conflicts = []
for a, b in product(courses_list, repeat=2):
    if a >= b: continue
    conflict = False
    for k1 in LTP:
        for k2 in LTP:
            for i, j in product(range(3), range(3)):
                day_a, _, end_a = y[a][k1][i]
                day_b, _, end_b = y[b][k2][j]
                if (day_a == day_b and day_a 
                    and not (end_a <= y[b][k2][j][1] or end_b <= y[a][k1][i][1])):
                    conflict = True
                    break
            if conflict: break
        if conflict: break
    if conflict:
        time_conflicts.append((a, b))

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
df2 = pd.read_csv("data/major_requirements.csv")
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
for a, b in time_conflicts:
    for s in semesters_list:
        model.Add(x[(a,s)] + x[(b,s)] <= 1)

#Hard Constraint 6(Electives)
eldf = pd.read_csv("data/electives_catalog.csv")

for s in range(5,9):
    model.Add(sum(x[(c,s)] for c in eldf["open_electives"].to_list()) <= 1)
for s in range(5,9):
    model.Add(sum(x[(c,s)] for c in eldf[major_pref].to_list()) <= 1)

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

penalty_timings = model.NewIntVar(0, 10_000_000, "penalty_timings")
model.Add(penalty_timings == penalty_morn + penalty_eve)

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


penalty_gaps = model.NewIntVar(0, 10_000_000, "penalty_gaps")
model.Add(penalty_gaps == sum(gap_penalty_vars))


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
    
imbalance = model.NewIntVar(0, 10_000_000, "imbalance")
model.Add(imbalance == sum(day_diff_vars))

#5. Irregularity
extra_course_bonus = []
for s in semesters_list:
    num_courses = sum(x[(c,s)] for c in courses_list)
    extra = model.NewIntVar(0, 2, f"extra_courses_s{s}")
    model.Add(extra == num_courses - 3)
    extra_course_bonus.append(extra)
penalty_irregularity = sum(extra_course_bonus)

model.Minimize(penalty_workload + penalty_timings + penalty_gaps - penalty_irregularity + imbalance)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0   
solver.parameters.log_search_progress = True
solver.Solve(model)
#store baseline solution
x0 = {(c,s): solver.Value(x[(c,s)]) for c in courses_list for s in semesters_list}

#triggers for reoptimization
def perf_feas():
    if solver.Value(penalty_workload)>10000 or solver.Value(penalty_timings)>10000 or solver.Value(penalty_gaps)>10000 or solver.Value(imbalance)>10000:
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
        robust_model.Add(sum(x[(c, s)] for c in courses_list) >= 4)

    #Hard Constraint 2(required courses for a particular major must be taken exactly once)
    req_string = str(df2[df2["major"] == major_pref]["required_courses"].values[0])
    for c in req_string.split("_") :
        robust_model.Add(sum(x[(c,s)] for s in semesters_list) == 1)

    #Hard Constraint 3(credit limits)
    semester_credits = {}
    for s in semesters_list:
        semester_credits[s] = sum(x[(c,s)] * int(credits_dict[c]) for c in courses_list)
    
    for s in semesters_list:
        robust_model.Add(semester_credits[s] <= max_credits)
        robust_model.Add(semester_credits[s] >= min_credits)
    

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
    for a, b in time_conflicts:
        for s in semesters_list:
            robust_model.Add(x[(a,s)] + x[(b,s)] <= 1)

    #Hard Constraint 6(Electives)

    for s in range(5,9):
        robust_model.Add(sum(x[(c,s)] for c in eldf["open_electives"].to_list()) <= 1)
    for s in range(5,9):
        robust_model.Add(sum(x[(c,s)] for c in eldf[major_pref].to_list()) <= 1)

    #Hard Constraint 7(Co-requisites)
    for c in courses_list:
        for s in semesters_list:
                must = df[df["course_name"] == c]["corequisites"].values[0].split("|")
                for m in must:
                    if m == "NONE":
                        continue
                    else:
                        robust_model.Add(x[(m,s)] >= x[(c,s)])

    #Hard Constraint 8(Semester Availabilty)
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

    penalty_timings = robust_model.NewIntVar(0, 10_000_000, "penalty_timings")
    robust_model.Add(penalty_timings == penalty_morn + penalty_eve)

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


    penalty_gaps = robust_model.NewIntVar(0, 10_000_000, "penalty_gaps")
    robust_model.Add(penalty_gaps ==  sum(gap_penalty_vars))


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
        
    imbalance = robust_model.NewIntVar(0, 10_000_000, "imbalance")
    robust_model.Add(imbalance == sum(day_diff_vars))

    #Robust model requirements
    for c in courses_list:
        for s in semesters_list:
            if int(s) < int(current_semester) and x0[(c,s)] == 1:
                robust_model.Add(x[(c, s)] == x0[(c,s)])


    delta = {}

    for c in courses_list:
        for s in semesters_list:
            delta[(c,s)] = robust_model.NewIntVar(0,6,f"delta_{c}_{s}")
            robust_model.Add(delta[(c,s)] >= x[(c,s)] - x0[(c,s)])
            robust_model.Add(delta[(c,s)] >= x0[(c,s)] - x[(c,s)])
    penalty_stability = robust_model.NewIntVar(0, 10_000_000, "penalty_stability")
    robust_model.Add(penalty_stability == sum((9 - s) * delta[(c, s)] for c in courses_list for s in semesters_list))

    #5. Irregularity
    extra_course_bonus = []
    for s in semesters_list:
        num_courses = sum(x[(c,s)] for c in courses_list)
        extra = robust_model.NewIntVar(0, 2, f"extra_courses_s{s}")
        robust_model.Add(extra == num_courses - 3)
        extra_course_bonus.append(extra)
    penalty_irregularity = sum(extra_course_bonus)

    robust_solver = cp_model.CpSolver()

    robust_model.Minimize(800*penalty_workload)
    robust_solver.parameters.max_time_in_seconds = 30
    robust_solver.parameters.log_search_progress = True
    robust_solver.Solve(robust_model)
    robust_model.Add(penalty_workload == robust_solver.Value(penalty_workload))

    robust_model.Minimize(500*penalty_timings)
    robust_solver.parameters.max_time_in_seconds = 30
    robust_solver.parameters.log_search_progress = True
    robust_solver.Solve(robust_model)
    robust_model.Add(penalty_timings == robust_solver.Value(penalty_timings))

    robust_model.Maximize(500*penalty_irregularity)
    robust_solver.parameters.max_time_in_seconds = 30
    robust_solver.parameters.log_search_progress = True
    robust_solver.Solve(robust_model)
    robust_model.Add(penalty_irregularity == robust_solver.Value(penalty_irregularity))

    robust_model.Minimize(200*penalty_gaps)
    robust_solver.parameters.max_time_in_seconds = 30
    robust_solver.parameters.log_search_progress = True
    robust_solver.Solve(robust_model)
    robust_model.Add(penalty_gaps == robust_solver.Value(penalty_gaps))

    robust_model.Minimize(100*imbalance)
    robust_solver.parameters.max_time_in_seconds = 30
    robust_solver.parameters.log_search_progress = True
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
    solver2 = cp_model.CpSolver()

    model.Minimize(800*penalty_workload)
    solver2.parameters.max_time_in_seconds = 30
    solver2.parameters.log_search_progress = True
    solver2.Solve(model)
    model.Add(penalty_workload == solver2.Value(penalty_workload))

    model.Minimize(500*penalty_timings)
    solver2.parameters.max_time_in_seconds = 30
    solver2.parameters.log_search_progress = True
    solver2.Solve(model)
    model.Add(penalty_timings== solver2.Value(penalty_timings))
    
    model.Maximize(500*penalty_irregularity)
    solver2.parameters.max_time_in_seconds = 30
    solver2.parameters.log_search_progress = True
    solver2.Solve(model)
    model.Add(penalty_irregularity==solver2.Value(penalty_irregularity))
    
    model.Minimize(200*penalty_gaps)
    solver2.parameters.max_time_in_seconds = 30
    solver2.parameters.log_search_progress = True
    solver2.Solve(model)
    model.Add(penalty_gaps == solver2.Value(penalty_gaps))

    model.Minimize(100*imbalance)
    solver2.parameters.max_time_in_seconds = 30
    solver2.parameters.log_search_progress = True
    status = solver2.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Baseline model infeasible")
        exit()

    print("Baseline Solution")
    print(
        f"workload={solver2.Value(penalty_workload)} "
        f"timings={solver2.Value(penalty_timings)} "
        f"gaps={solver2.Value(penalty_gaps)} "
        f"imbalance={solver2.Value(imbalance)}"
    )

    for s in semesters_list:
        courses = [
            c for (c, sem), v in x.items()
            if sem == s and solver2.Value(v) == 1
        ]
        if courses:
            print(f"Semester {s}: {courses}")