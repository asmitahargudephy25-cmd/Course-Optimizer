import pandas as pd
df = pd.read_csv("course_catalog.csv")
courses_list = list(df["course_name"])
semesters_list = [1,2,3,4,5,6,7,8]
reqcourses_list = list(df[df['required_course'] == "YES"]["course_name"])
credits_list = list(df["credits"])

from ortools.sat.python import cp_model 

#building model
model = cp_model.CpModel()

#creating decision variables(each variable has a domain{0,1})
x = {}
for c in courses_list:
    for s in semesters_list:
        x[('c','s')] = model.new_bool_var(f"x_{c}_{s}")

#Hard Constraint 1(each course must be taken atmost once)
for c in courses_list:
    model.add(sum(x[('c','s')] for s in semesters_list) <= 1)

#Hard Constraint 2(required courses must be taken exactly once)
for courses in reqcourses_list:
    model.add(sum(x[('c','s')] for s in semesters_list) == 1)

#Hard Constraint 3(credit limits)


   
                  





