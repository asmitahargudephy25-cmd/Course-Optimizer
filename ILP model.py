import pandas as pd
df = pd.read_csv("course_catalog.csv")
courses_list = list(df["course_name"])
semesters_list = [1,2,3,4,5,6,7,8]
reqcourses_list = list(df[df['required_course'] == "YES"]["course_name"])
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
    model.add(sum(x[(c,s)] for s in semesters_list) <= 1)

#Hard Constraint 2(required courses must be taken exactly once)
for courses in reqcourses_list:
    model.add(sum(x[(c,s)] for s in semesters_list) == 1)

#Hard Constraint 3(credit limits)
for s in semesters_list:
    model.add(sum(x[(c,s)]*credits_dict[c] for c in courses_list) <= max_credits)
for s in semesters_list:
    model.add(sum(x[(c,s)]*credits_dict[c] for c in courses_list) >= min_credits)

#Hard Constraint 4
new_df = df[df["prerequisites"] != "NONE"]
prereq_dict = new_df.set_index("course_name")["prerequisites"].to_dict()
for k in prereq_dict:
    prereq_dict[k] = prereq_dict[k].split("_")

for key in prereq_dict.keys():
    for p in prereq_dict[key]:
        for s in semesters_list:
            model.add(x[key,s]<= sum(x[p,t] for t in range(1,s)))


#solver
solver = cp_model.CpSolver()
result = solver.Solve(model)


#output
if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("Optimal Schedule:")
    for s in semesters_list:
        taken = [c for c in courses_list if solver.Value(x[(c, s)]) == 1]
        if taken:
            print(f"Semester {s+1}: {taken}")
else:
    print("No feasible schedule found.")
