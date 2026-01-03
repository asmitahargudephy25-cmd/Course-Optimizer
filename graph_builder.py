import networkx as nx
import pandas as pd

class CourseGraphBuilder:
    def __init__(self):
        self.g = nx.DiGraph()

    def AddCourse(self,course,credits,prerequsites,corequisites,difficulty,semesters_aval,lecture,tutorial,practical):
        self.g.add_node(course,credits = credits,prerequsites = prerequsites, corequisites = corequisites,
                   difficulty = difficulty,availability = semesters_aval,
                   lecture = lecture,tutorial = tutorial,practical = practical)
        
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
for index,row in df.iterrows():
    graph.AddCourse(row["course_name"],row["credits"],str(row["prerquisites"]).split("_"),
                    str(row["coerquisites"]).split("_"),row["difficulty"],
                    str(row["semesters_available"]).split("|"),)
    
