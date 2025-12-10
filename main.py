import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---

class TaskPauseLog(BaseModel):
    id: int
    session_id: int
    pause_start: str
    pause_end: Optional[str] = None
    duration: Optional[int] = None
    created_at: str

class TaskSession(BaseModel):
    id: int
    task_id: int
    start_time: str
    end_time: Optional[str] = None
    active_duration: int
    status: str
    created_at: str
    task_pause_logs: List[TaskPauseLog] = []

class Task(BaseModel):
    id: int
    title: str
    date: str
    planned_duration: int
    status: str
    created_at: str
    task_sessions: List[TaskSession] = []

    # Handle 'title' as 'string' type in schema but python calls it 'str'
    # Wait, 'string' is not a valid python type, it's 'str'. Correcting below.

class TaskModel(BaseModel):
    id: int
    title: str
    date: str
    planned_duration: int
    status: str
    created_at: str
    task_sessions: List[TaskSession] = []

class DailySummaryModel(BaseModel):
    id: Optional[int] = None
    date: str
    total_focus_time: int
    total_pause_time: int
    distractions_count: int
    created_at: Optional[str] = None

class DailyInsightRequest(BaseModel):
    date: str
    summary: DailySummaryModel
    tasks: List[TaskModel]

class WeeklyInsightRequest(BaseModel):
    start_date: str
    end_date: str
    daily_summaries: List[DailySummaryModel]
    tasks: List[TaskModel]

class TaskInsightRequest(BaseModel):
    task: TaskModel

# Response Models (matching frontend interfaces)

class TaskInsightDetail(BaseModel):
    task_id: str
    study_minutes: float
    pauses: int

class DailyAIInsightResponse(BaseModel):
    date: str
    total_study_minutes: float
    distractions: int
    pauses: int
    max_focus_block: float
    min_focus_block: float
    avg_focus_block: float
    peak_focus_hour: str
    slump_hour: str
    productivity_score: int
    focus_efficiency: float
    tips: List[str]
    tasks: List[TaskInsightDetail]

class DailyStatDetail(BaseModel):
    date: str
    study_minutes: float
    distractions: int

class WeeklyAIInsightResponse(BaseModel):
    week_start: str
    total_study_minutes: float
    total_distractions: int
    avg_daily_study: float
    avg_daily_distractions: float
    best_focus_day: str
    worst_focus_day: str
    max_focus_block_week: float
    weekly_productivity_score: int
    consistency_score: int
    weekly_tips: List[str]
    daily: List[DailyStatDetail]

# --- Helper Functions ---

def calculate_daily_stats(tasks: List[TaskModel], summary: DailySummaryModel):
    # Calculate stats from raw data
    total_study_minutes = summary.total_focus_time / 60
    distractions = summary.distractions_count
    pauses = summary.total_pause_time // 60 # approx count or minutes? schema says 'total_pause_time' in seconds.
    # Frontend 'pauses' in DailyAIInsight seems to be a count or minutes? 
    # In insightApi.ts: "pauses: number". Usually count.
    # But 'total_pause_time' is duration.
    # Let's count pauses from task_pause_logs if available, or just use distraction count as proxy?
    # Actually, we can count total sessions or pauses in sessions.
    
    pause_count = 0
    focus_blocks = []
    task_details = []
    
    hourly_focus = {} # "HH" -> minutes

    for task in tasks:
        task_study_minutes = 0
        task_pauses = 0
        
        for session in task.task_sessions:
            # Active duration is in seconds
            duration_min = session.active_duration / 60
            if duration_min > 0:
                focus_blocks.append(duration_min)
                task_study_minutes += duration_min
                
                # Attribute to hour
                start_dt = datetime.fromisoformat(session.start_time.replace('Z', '+00:00'))
                hour = start_dt.strftime("%H:00")
                hourly_focus[hour] = hourly_focus.get(hour, 0) + duration_min

            task_pauses += len(session.task_pause_logs)
            pause_count += len(session.task_pause_logs)
        
        task_details.append(TaskInsightDetail(
            task_id=str(task.id),
            study_minutes=round(task_study_minutes, 1),
            pauses=task_pauses
        ))

    max_focus = max(focus_blocks) if focus_blocks else 0
    min_focus = min(focus_blocks) if focus_blocks else 0
    avg_focus = sum(focus_blocks) / len(focus_blocks) if focus_blocks else 0
    
    # Peak hour
    peak_hour = max(hourly_focus, key=hourly_focus.get) if hourly_focus else "N/A"
    slump_hour = min(hourly_focus, key=hourly_focus.get) if hourly_focus else "N/A" 
    # (Note: slump might be hour with 0, but we only track hours with activity. 
    # Real slump is complex, but let's just take min of active hours for now or "15:00" default)
    
    # Efficiency: Focus time / (Focus + Pause)
    total_time = summary.total_focus_time + summary.total_pause_time
    efficiency = (summary.total_focus_time / total_time * 100) if total_time > 0 else 0

    return {
        "total_study_minutes": round(total_study_minutes, 1),
        "distractions": distractions,
        "pauses": pause_count, # Using calculated pause count from logs
        "max_focus_block": round(max_focus, 1),
        "min_focus_block": round(min_focus, 1),
        "avg_focus_block": round(avg_focus, 1),
        "peak_focus_hour": peak_hour,
        "slump_hour": slump_hour,
        "focus_efficiency": round(efficiency, 1),
        "task_details": task_details
    }

# --- LangChain Logic ---

def get_daily_ai_analysis(stats: dict):
    # Use Groq to generate tips and score
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"tips": ["Add GROQ_API_KEY to .env for AI tips."], "score": 50}
    
    chat = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-70b-8192")
    
    system = "You are a productivity expert. Analyze the daily stats and provide: 1. A productivity score (0-100). 2. Three specific, actionable tips."
    human = f"Stats: {json.dumps(stats)}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    
    # Define structure for output parsing (or just regex it)
    # For simplicity, let's ask for JSON
    
    json_schema = {
        "score": "integer",
        "tips": ["string", "string", "string"]
    }
    
    prompt_with_format = ChatPromptTemplate.from_messages([
        ("system", system + " Respond in JSON format: " + json.dumps(json_schema)),
        ("human", human)
    ])
    
    chain = prompt_with_format | chat | StrOutputParser()
    
    try:
        response = chain.invoke({})
        # Extract JSON from response (handle potential markdown blocks)
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json")[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.split("```")[0]
        
        data = json.loads(cleaned)
        return data
    except Exception as e:
        print(f"Error calling AI: {e}")
        return {"tips": ["Stay focused!", "Take breaks.", "Plan ahead."], "score": 75}

def get_weekly_ai_analysis(stats: dict):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"tips": ["Add GROQ_API_KEY to .env"], "productivity_score": 50, "consistency_score": 50}
        
    chat = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-70b-8192")
    
    system = "You are a productivity expert. Analyze weekly stats. Provide: 1. Weekly productivity score (0-100). 2. Consistency score (0-100). 3. Three strategic tips for next week."
    human = f"Stats: {json.dumps(stats)}"
    
    json_schema = {
        "productivity_score": "integer",
        "consistency_score": "integer",
        "tips": ["string", "string", "string"]
    }
    
    prompt_with_format = ChatPromptTemplate.from_messages([
        ("system", system + " Respond in JSON format: " + json.dumps(json_schema)),
        ("human", human)
    ])
    
    chain = prompt_with_format | chat | StrOutputParser()
    
    try:
        response = chain.invoke({})
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json")[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.split("```")[0]
            
        data = json.loads(cleaned)
        return data
    except Exception as e:
        print(f"Error calling AI: {e}")
        return {"tips": ["Review your week.", "Plan next week.", "Rest well."], "productivity_score": 70, "consistency_score": 70}

def get_task_ai_analysis(task_data: dict):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Focus on your task!"
    
    chat = ChatGroq(temperature=0.7, groq_api_key=api_key, model_name="llama3-70b-8192")
    
    system = "You are a productivity coach. Provide a short, encouraging insight or tip for this specific task."
    human = f"Task: {json.dumps(task_data)}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | chat | StrOutputParser()
    
    try:
        return chain.invoke({})
    except Exception:
        return "Break it down into smaller steps!"


# --- Routes ---

@app.post("/ai_daily_insight", response_model=DailyAIInsightResponse)
async def generate_daily_insight(request: DailyInsightRequest):
    stats = calculate_daily_stats(request.tasks, request.summary)
    
    # Simplify stats for AI to save tokens/avoid noise
    ai_input = {k: v for k, v in stats.items() if k != "task_details"}
    
    ai_result = get_daily_ai_analysis(ai_input)
    
    return DailyAIInsightResponse(
        date=request.date,
        total_study_minutes=stats["total_study_minutes"],
        distractions=stats["distractions"],
        pauses=stats["pauses"],
        max_focus_block=stats["max_focus_block"],
        min_focus_block=stats["min_focus_block"],
        avg_focus_block=stats["avg_focus_block"],
        peak_focus_hour=stats["peak_focus_hour"],
        slump_hour=stats["slump_hour"],
        productivity_score=ai_result.get("score", 70),
        focus_efficiency=stats["focus_efficiency"],
        tips=ai_result.get("tips", []),
        tasks=stats["task_details"]
    )

@app.post("/ai_task_insight")
async def generate_task_insight(request: TaskInsightRequest):
    task_data = {
        "title": request.task.title,
        "duration": request.task.planned_duration,
        "status": request.task.status
    }
    insight = get_task_ai_analysis(task_data)
    return {"insight_text": insight}

@app.post("/weekly_analytics", response_model=WeeklyAIInsightResponse)
async def generate_weekly_insight(request: WeeklyInsightRequest):
    # Calculate weekly stats
    total_minutes = 0
    total_distractions = 0
    daily_stats = []
    
    # Map summaries by date
    summary_map = {s.date: s for s in request.daily_summaries}
    
    # Iterate over request days (or summaries)
    # We should iterate over the range, but let's just use the provided summaries
    
    for summary in request.daily_summaries:
        minutes = summary.total_focus_time / 60
        total_minutes += minutes
        total_distractions += summary.distractions_count
        daily_stats.append(DailyStatDetail(
            date=summary.date,
            study_minutes=round(minutes, 1),
            distractions=summary.distractions_count
        ))
        
    num_days = len(request.daily_summaries) if request.daily_summaries else 1
    avg_daily_study = total_minutes / num_days
    avg_daily_distractions = total_distractions / num_days
    
    # Best/Worst days
    if daily_stats:
        best_day = max(daily_stats, key=lambda x: x.study_minutes).date
        worst_day = min(daily_stats, key=lambda x: x.study_minutes).date
    else:
        best_day = "N/A"
        worst_day = "N/A"
        
    # Max focus block in week
    # Need to iterate all tasks
    max_block = 0
    for task in request.tasks:
        for session in task.task_sessions:
            duration_min = session.active_duration / 60
            if duration_min > max_block:
                max_block = duration_min
                
    stats_for_ai = {
        "total_minutes": total_minutes,
        "total_distractions": total_distractions,
        "avg_daily_study": avg_daily_study,
        "best_day": best_day,
        "worst_day": worst_day
    }
    
    ai_result = get_weekly_ai_analysis(stats_for_ai)
    
    return WeeklyAIInsightResponse(
        week_start=request.start_date,
        total_study_minutes=round(total_minutes, 1),
        total_distractions=total_distractions,
        avg_daily_study=round(avg_daily_study, 1),
        avg_daily_distractions=round(avg_daily_distractions, 1),
        best_focus_day=best_day,
        worst_focus_day=worst_day,
        max_focus_block_week=round(max_block, 1),
        weekly_productivity_score=ai_result.get("productivity_score", 70),
        consistency_score=ai_result.get("consistency_score", 70),
        weekly_tips=ai_result.get("tips", []),
        daily=daily_stats
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
