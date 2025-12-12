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
def normalize_and_pad(slot_str):
    # Split the slot string into individual slots (ignore "NONE")
    if slot_str == "NONE":
        slot_list = []
    else:
        slot_list = [slot.split("_") for slot in slot_str.split("/")]

    # Normalize slot to 3 elements each
    normalized = [
        tuple(parts + [None] * (3 - len(parts)))   # pad inside tuple
        for parts in slot_list
    ]

    # Pad L/T/P to always have exactly 3 slots
    while len(normalized) < 3:
        normalized.append((None, None, None))

    # Trim in case there are more than 3 slots
    return tuple(normalized[:3])


y = {}

for c in courses_list:
    row = df[df["course_name"] == c].iloc[0]

    y[c] = {
        "l": normalize_and_pad(row["lecture"]),
        "t": normalize_and_pad(row["tutorial"]),
        "p": normalize_and_pad(row["practical"])
    }
    
LTP = ("l", "t", "p")

for a in courses_list: 
    for b in courses_list: 
        for i in range(3): 
            for j in range(3): 
                for k in LTP: 
                    for m in LTP: 
                        if y[a][m][i][0] == y[b][k][j][0] and m!=k and i!=j and a!=b and y[a][m][i][0] != None: 
                            if y[a][m][i][1] < y[b][k][j][2] or y[b][k][j][1] < y[a][m][i][2] or y[a][m][i][1] == y[b][k][j][1]: 
                                model.Add(x[a,s] + x[b,s] <= 1)

#Hard Constraint 5(Electives)
eldf = pd.read_csv("electives_catalog.csv")

for s in range(5,9):
    model.Add(sum(x[(c,s)] for c in eldf["open_electives"].to_list()) == 1)
for s in range(5,9):
    model.Add(sum(x[(c,s)] for c in eldf[major_pref].to_list()) == 1)

#Must Optimise objective(workload variance across all semesters):
workload = {}
for s in semesters_list:
    workload[s] = sum(x[(c, s)]*int(df[df["course_name"] == c]["difficulty"]) for c in courses_list)
diff = {}
for i in semesters_list:
    for j in semesters_list:
        if i < j:
            d = model.new_int_var(0, 50, f"diff_{i}_{j}")
            diff[(i,j)] = d
            model.Add(d >= workload[i] - workload[j])
            model.Add(d >= workload[j] - workload[i])
model.Minimize(sum(diff.values()))

#Pareto optimisation

#1a.Morning classes(before 10)
for c in courses_list:
    for a in LTP:
        for b in range(3):
            if y[c][a][b][1] != None and int(y[c][a][b][1]) <= 10:
                penalty_morn = sum(x[(c, s)]*10 for s in semesters_list)

#1b.Classes after 5
for c in courses_list:
    for a in LTP:
        for b in range(3):
            if y[c][a][b][1] != None and int(y[c][a][b][1]) >= 5:
                penalty_eve = sum(x[(c, s)]*7 for s in semesters_list)

#1c.Minimize gaps
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

total_gaps = 0

for day, items in day_slots.items():
    items.sort(key=lambda x: x[0])
    for i in range(len(items) - 1):
        end_current = items[i][1]
        start_next = items[i+1][0]
        if start_next > end_current:
            total_gaps += (int(start_next) - int(end_current))
penalty_gaps = total_gaps*5

penalty = penalty_eve + penalty_morn + penalty_gaps

#2.Fairness Imbalance
keys = list(day_slots.keys())
pairs = []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        pairs.append((keys[i], keys[j]))

diff = 0
for day1,day2 in pairs:
    if len(day1) > len(day2):
        diff = diff + len(day1) - len(day2)
    else:
        diff = diff + len(day2) - len(day1)



def extract_schedule(mod):
    solver = cp_model.CpSolver()
    if solver.Solve(mod) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Optimal Schedule:")
    for s in semesters_list:
        taken = [c for c in courses_list if solver.Value(x[(c, s)]) == 1]
        if taken:
            print(f"Semester {s}: {taken}")
        else:
            print("No feasible schedule found.")        

solver = cp_model.CpSolver()

solver.Solve(model)
extract_schedule(model)
print(f"penalty = {solver.Value(penalty)}")
print(f"imbalance = {solver.Value(diff)}")
""""
model_b = model.Copy()
solver.Solve(model_b.Minimize(penalty))
model_b.Add(penalty == solver.Value(penalty))
solver.Solve(model_b.Minimise(diff))
extract_schedule(model_b)
print(penalty = solver.Value(penalty))
print(imbalance = solver.Value(diff))
"""