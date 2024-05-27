import os
from textwrap import dedent
from crewai import Agent
from langchain_groq import ChatGroq

from tools import ExaSearchToolset

class MeetingPrepAgents():
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="mixtral-8x7b-32768"
        )

    def research_agent(self):
        return Agent(
            role='Research Specialist',
            goal='Conduct thorough research on people and companies involved in the meeting.',
            tools=ExaSearchToolset.tools(),
            backstory=dedent("""\
                As a Research Specialist, your mission is to uncover detailed infromation
                about the individuals and entities participating in the meeting. Your insights
                will lay the groundwork for strategic meeting preparation.
                             """),
            llm=self.llm,
            max_iter=2,
            verbose=True
        )
    
    def industry_analysis_agent(self):
        return Agent(
            role='Industry Analyst',
            goal='Analyze the current industry trends, challenges, and opportunities.',
            tools=ExaSearchToolset.tools(),
            backstory=dedent("""\
                As an Industry Analyst, your analysis will identify key trends,
                challenges facing the industry, and potential opportunities that
                could be leveraged during the meeting for strategic advantage.
                             """),
            llm=self.llm,
            max_iter=2,
            verbose=True
        )
    
    def meeting_strategy_agent(self):
        return Agent(
            role='Meeting Strategy Advisor',
            goal='Develop talking points, questions, and strategic angles for the meeting.',
            backstory=dedent("""\
                As a strategy Advisor, your expertise will guide the development of
                talking points, insightful questions, and strategic angles
                to ensure the meeting's objectives are acived.
                             """),
            llm=self.llm,
            max_iter=2,
            verbose=True
        )
    
    def summary_and_briefing_agent(self):
        return Agent(
            role='Briefing coordinator',
            goal='Compile all gathered information into a concise, informative briefing document.',
            backstory=dedent("""\
                As the Briefing Coordinator, your role is to consolidate the research,
                analysis, and strategic insights.
                             """),
            llm=self.llm,
            max_iter=2,
            verbose=True
        )