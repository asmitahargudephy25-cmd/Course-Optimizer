import pandas as pd
df = pd.read_csv("course_catalog.csv")
courses_list = list(df["course_name"])
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
major_pref = input("Your preferred major:")
df2 = pd.read_csv("major_requirements.csv")

req_string = df2[df2["major"] == major_pref]["required_courses"].values[0]
for c in req_string.split("_") :
    model.Add(sum(x[(c,s)] for s in semesters_list) == 1)

#Hard Constraint 3(credit limits)
total_credits = sum(x[(c,s)]*int(credits_dict[c]) for c in courses_list)
for s in semesters_list:
    model.Add(total_credits <= max_credits)
for s in semesters_list:
    model.Add(total_credits >= min_credits)

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
y = {}
for c in courses_list:
    y[c] = z

    z = {"l" : (tuple(str(df["lecture"].values[0]).split("/"))[i].split("_") for i in range(3)),
         "t" : (tuple(str(df["tutorial"].values[0]).split("/"))[i].split("_") for i in range(3)),
         "p" : (tuple(str(df["tutorial"].values[0]).split("/"))[i].split("_") for i in range(3))
 }

LTP = ("l", "t" ,"p")
for a in courses_list:
    for b in courses_list:
        for i in range(3):
            for j in range(3):
                for k in LTP:
                    for m in LTP:
                        if y[a][m][i][0] == y[b][k][j][0] and m!=k and i!=j and a!=b:
                            if y[a][m][i][1] < y[b][k][j][2] or y[b][k][j][1] < y[a][m][i][2] or y[a][m][i][1] == y[b][k][j][1]:
                                model.Add(x[a,s] + x[b,s] <= 1)






#solver
solver = cp_model.CpSolver()
result = solver.Solve(model)


#output
if result in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("Optimal Schedule:")
    for s in semesters_list:
        taken = [c for c in courses_list if solver.Value(x[(c, s)]) == 1]
        if taken:
            print(f"Semester {s}: {taken}")
else:
    print("No feasible schedule found.")
