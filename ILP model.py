import pandas as pd
df = pd.read_csv("course_catalog.csv")
courses_list = df["course_name"].to_list()
semesters_list = [1,2,3,4,5,6,7,8]
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
        x[(c,s)] = model.new_bool_var(f"x_{c}_{s}")

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
for s in semesters_list:
    semester_credits = sum(x[(c,s)] * int(credits_dict[c]) for c in courses_list)
    model.Add(semester_credits <= max_credits)
    model.Add(semester_credits >= min_credits)


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

#Hard Constraint 5(Electives)
eldf = pd.read_csv("electives_catalog.csv")

for s in range(5,9):
    model.Add(sum(x[(c,s)] for c in eldf["open_electives"].to_list()) == 1)
for s in range(5,9):
    model.Add(sum(x[(c,s)] for c in eldf[major_pref].to_list()) == 1)

#Hard Constraint 6(Co-requisites)
for c in courses_list:
    for s in semesters_list:
            must = df[df["course_name"] == c]["corequisites"].values[0].split("|")
            for m in must:
                if m == "NONE":
                    continue
                else:
                    model.Add(x[(m,s)] >= x[(c,s)])

#Hard Constraint 7(Semester Availabilty)
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
    workload[s] = sum(x[(c, s)]*int(df[df["course_name"] == c]["difficulty"].iloc[0]) for c in courses_list)
diff = {}
for i in semesters_list:
    for j in semesters_list:
        if i < j:
            d = model.new_int_var(0, 50, f"diff_{i}_{j}")
            diff[(i,j)] = d
            model.Add(d >= workload[i] - workload[j])
            model.Add(d >= workload[j] - workload[i])
penalty_workload = sum(diff.values())


#2a.Morning classes(before 10)
penalty_morn = sum(x[(c, s)]*10 for s in semesters_list 
                                for c in courses_list 
                                for a in LTP 
                                for b in range(3)
                                if y[c][a][b][1] != None and int(y[c][a][b][1]) <= 10)

#2b.Classes after 5

penalty_eve = sum(x[(c, s)]*7 for s in semesters_list
                              for c in courses_list
                              for a in LTP
                              for b in range(3)
                              if y[c][a][b][1] != None and int(y[c][a][b][1]) >= 5)

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

gap_pairs = []

for day, items in day_slots.items():
    items.sort(key=lambda x: x[0])
    for i in range(len(items) - 1):
        c1 = items[i][2]
        c2 = items[i+1][2]
        gap = int(items[i+1][0]) - int(items[i][1])

        if gap > 0:
            gap_pairs.append((c1, c2, gap))

gap_penalty_vars = []

for (c1, c2, gap) in gap_pairs:
    for s in semesters_list:
        g = model.NewBoolVar(f"gap_{c1}_{c2}_{s}")

        # g = 1 if both courses are taken in semester s
        model.Add(g <= x[(c1, s)])
        model.Add(g <= x[(c2, s)])
        model.Add(g >= x[(c1, s)] + x[(c2, s)] - 1)

        gap_penalty_vars.append(gap * g)

penalty_gaps = 5*sum(gap_penalty_vars)


#4.Fairness Imbalance
day_workload = {}

for day in day_slots:
    day_workload[day] = sum(x[(c, s)] * int(df[df["course_name"] == c]["difficulty"].iloc[0]) for (start, end, c) in day_slots[day] for s in semesters_list)

day_diff_vars = []

days = list(day_workload.keys())

for i in range(len(days)):
    for j in range(i + 1, len(days)):
        d = model.new_int_var(0, 100, f"day_diff_{days[i]}_{days[j]}")
        model.Add(d >= day_workload[days[i]] - day_workload[days[j]])
        model.Add(d >= day_workload[days[j]] - day_workload[days[i]])
        day_diff_vars.append(d)
    
imbalance = 3*sum(day_diff_vars)

model.Minimize(penalty_workload + penalty_timimgs + penalty_gaps + imbalance)

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

solver = cp_model.CpSolver()
solver.parameters.enumerate_all_solutions = True
solver.parameters.max_time_in_seconds = 15
callback = ParetoCallback([penalty_workload, penalty_timimgs, penalty_gaps, imbalance],x)
solver.Solve(model, callback)

def is_dominated(sol, others):
    return any(
        all(o <= s for o, s in zip(other[0], sol[0])) and
        any(o < s for o, s in zip(other[0], sol[0]))
        for other in others
    )

pareto_solutions = [
    sol for sol in callback.solutions
    if not is_dominated(sol, callback.solutions)
]

for i, (obj, assign) in enumerate(pareto_solutions, 1):
    print(f"Solution {i}")
    print(f"workload={obj[0]} timing={obj[1]} gaps={obj[2]} imbalance={obj[3]}")
    for s in semesters_list:
        courses = [c for (c, sem) in assign if sem == s]
        if courses:
            print(f"Semester {s}: {courses}")
    print("*" * 40)

x# = {(c,s): solver.Value(x[(c,s)]) for c in courses_list for s in semesters_list}

