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

def course_conflict(c1,c2):
    slots1 = (graph.g.nodes[c1]["lecture"] + graph.g.nodes[c1]["tutorial"] + graph.g.nodes[c1]["practical"])
    slots2 = (graph.g.nodes[c2]["lecture"] + graph.g.nodes[c2]["tutorial"] + graph.g.nodes[c2]["practical"])

    for slot1 in slots1:
        for slot2 in slots2:
            if slot_conflict(slot1,slot2):
                return True
            
    return False

class CourseGraphBuilder:
    def __init__(self):
        self.g = nx.DiGraph()

    def AddCourse(self,course,credits,prerequsites,corequisites,difficulty,semesters_aval,lecture,tutorial,practical):
        self.g.add_node(course,credits = credits,prerequsites = prerequsites, corequisites = corequisites,
                   difficulty = difficulty,availability = semesters_aval,
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
courses_list = g.nodes.to_list()
graph = CourseGraphBuilder()
for index,row in df.iterrows():
    graph.AddCourse(row["course_name"],row["credits"],str(row["prerquisites"]).split("_"),
                    str(row["coerquisites"]).split("_"),row["difficulty"],
                    str(row["semesters_available"]).split("|"),row["lecture"],row["tutorial"],row["practical"])
    
for index,row in df.iterrows():
    for i,exy in enumerate(str(row["prerquisites"]).split("_")):
        graph.AddPrerequisite(exy,row["course_name"])

for i in range(len(courses_list)):
    for j in range(i+1,len(courses_list)):
        if course_conflict(courses_list[i],courses_list[j]):
            graph.AddConflict(courses_list[i],courses_list[j])

