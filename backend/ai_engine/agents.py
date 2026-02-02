"""
CrewAI Agent Definitions — 3 agents per PRD Section 3
"""
import os
from crewai import Agent
from ai_engine.tools.data_tools import csv_reader, growth_calculator
from ai_engine.tools.sarimax_tool import forecast_revenue

LLM_MODEL = os.getenv("CREWAI_LLM_MODEL", "groq/llama-3.1-8b-instant")


def create_data_analyst() -> Agent:
    """Agent 1: Senior Data Analyst — cleans data and calculates metrics."""
    return Agent(
        role="Data Analyst",
        goal="Clean raw data and calculate metrics. Be concise.",
        backstory="You extract Date and Revenue columns, fix missing values, and calculate MoM growth. Output only facts.",
        tools=[csv_reader, growth_calculator],
        llm=LLM_MODEL,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )


def create_forecaster() -> Agent:
    """Agent 2: Quantitative Forecaster — predicts future revenue."""
    return Agent(
        role="Lead Forecaster",
        goal="Predict next 3 months of revenue. Be concise.",
        backstory="You run SARIMAX forecasts. Only output numbers and confidence intervals.",
        tools=[forecast_revenue],
        llm=LLM_MODEL,
        verbose=False,
        allow_delegation=False,
        max_iter=5,
    )


def create_strategist() -> Agent:
    """Agent 3: Strategic Advisor — synthesizes data into actionable advice."""
    return Agent(
        role="Startup Consultant",
        goal="Give 3 short, actionable recommendations based on the data.",
        backstory="You give direct startup advice. No fluff. If revenue drops, suggest cuts. If growing, suggest reinvestment. Only reference provided data.",
        tools=[],
        llm=LLM_MODEL,
        verbose=False,
        allow_delegation=False,
        max_iter=2,
    )
