courses_list = ["math", "cs", "physics", "calc1", "calc2"]
semesters_list = [1,2,3,4,5,6,7,8]
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
   
                  





