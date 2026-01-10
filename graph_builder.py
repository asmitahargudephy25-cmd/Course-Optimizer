import networkx as nx
import pandas as pd

def normalize_and_pad(slot_str):
    if slot_str == "NONE":
        slots = []
    else:
        slots = [s.split("_") for s in slot_str.split("/")]

    slots = [tuple(p + [None] * (3 - len(p))) for p in slots]

    while len(slots) < 3:
        slots.append((None, None, None))

    return tuple(slots[:3])

def slot_conflict(slot1,slot2):
    d1,s1,e1 = slot1
    d2,s2,e2 = slot2
    if d1 == None or d2 == None:
        return False
    if d1 != d2:
        return False
    if int(s1)<int(e2) and int(s2)<int(e1):
        return True
    return False

def course_conflict(G,c1,c2):
    slots1 = (G.nodes[c1]["lecture"] + G.nodes[c1]["tutorial"] + G.nodes[c1]["practical"])
    slots2 = (G.nodes[c2]["lecture"] + G.nodes[c2]["tutorial"] + G.nodes[c2]["practical"])

    for slot1 in slots1:
        for slot2 in slots2:
            if slot_conflict(slot1,slot2):
                return True
            
    return False

class CourseGraphBuilder:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def AddCourse(self,course,credits,prerequsites,corequisites,difficulty,semesters_avalaible,lecture,tutorial,practical):
        self.g.add_node(course,credits = credits,prerequsites = prerequsites, corequisites = corequisites,
                   difficulty = difficulty,availability = semesters_avalaible,
                   lecture = normalize_and_pad(lecture),tutorial = normalize_and_pad(tutorial),practical = normalize_and_pad(practical))
        
    def AddPrerequisite(self,c1,c2):
        self.g.add_edge(c1,c2,type = "prerequisite")

    def AddCorequisite(self,c1,c2):
        self.g.add_edge(c1,c2,type = "corequisite")
        self.g.add_edge(c2,c1,type = "corequisite")

    def AddConflict(self,c1,c2):
        self.g.add_edge(c1,c2,type = "conflict")
        self.g.add_edge(c2,c1,type = "conflict")

    def AddCreditLimits(self,max_credits,min_credits):
        self.g.graph["max_credits"] = max_credits
        self.g.graph["min_credits"] = min_credits

df = pd.read_csv("course_catalog.csv")
graph = CourseGraphBuilder()

for index, row in df.iterrows():
    availability = list(map(int, str(row["semesters_available"]).split("|")))

    graph.AddCourse(row["course_name"],row["credits"],str(row["prerequisites"]).split("_"),
                    str(row["corequisites"]).split("_"),row["difficulty"],
                    availability,row["lecture"],row["tutorial"],row["practical"])
    
courses_list = list(graph.g.nodes)
    
for index,row in df.iterrows():
    if str(row["prerequisites"]) != "NONE":
        for exy in str(row["prerequisites"]).split("_"):
            graph.AddPrerequisite(exy, row["course_name"])

for i in range(len(courses_list)):
    for j in range(i+1,len(courses_list)):
        if course_conflict(graph.g,courses_list[i],courses_list[j]):
            graph.AddConflict(courses_list[i],courses_list[j])

for index,row in df.iterrows():
    if str(row["corequisites"]) != "NONE":
        for co in str(row["corequisites"]).split("_"):
            graph.AddCorequisite(row["course_name"], co)


df2 = pd.read_csv("major_requirements.csv")
graph.g.graph["majors"]= {}
graph.g.graph["majors"]["required"] = {}
for index,row in df2.iterrows():
    graph.g.graph["majors"]["required"][row["major"]] = str(row["required_courses"]).split("_")

df3 = pd.read_csv("electives_catalog.csv")
graph.g.graph["open_electives"] = df3["open_electives"].to_list()
graph.g.graph["majors"]["electives"] = {}
for col_name,col_data in df3.items():
    graph.g.graph["majors"]["electives"][col_name] = col_data.to_list()