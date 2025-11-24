import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal, List, Optional
from dataclasses import dataclass, field, asdict

print("\n" + "🌿" * 50)
print("💡 agent.py LOADED SUCCESSFULLY!")
print("🌿" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 🧠 STATE MANAGEMENT & DATA STRUCTURES
# ======================================================

@dataclass
class CheckInState:
    """🌿 Holds data for the CURRENT daily check-in"""
    mood: str | None = None
    energy: str | None = None
    objectives: list[str] = field(default_factory=list)
    advice_given: str | None = None

    def is_complete(self) -> bool:
        """✅ Check if we have the core check-in data"""
        return all([
            self.mood is not None,
            self.energy is not None,
            len(self.objectives) > 0,
        ])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Userdata:
    """👤 User session data passed to the agent"""
    current_checkin: CheckInState
    history_summary: str  # String containing info about previous sessions
    session_start: datetime = field(default_factory=datetime.now)


# ======================================================
# 💾 PERSISTENCE LAYERS (JSON LOGGING)
# ======================================================
WELLNESS_LOG_FILE = "wellness_log.json"


def get_log_path() -> str:
    # Use __file__ (double underscores) – _file_ will crash.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(base_dir, ".."))
    return os.path.join(backend_dir, WELLNESS_LOG_FILE)


def load_history() -> list:
    """📖 Read previous check-ins from JSON"""
    path = get_log_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"⚠ Could not load history: {e}")
        return []


def save_checkin_entry(entry: CheckInState) -> None:
    """💾 Append new check-in to the JSON list"""
    path = get_log_path()
    history = load_history()

    # Create record
    record = {
        "timestamp": datetime.now().isoformat(),
        "mood": entry.mood,
        "energy": entry.energy,
        "objectives": entry.objectives,
        "summary": entry.advice_given,
    }

    history.append(record)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    print(f"\n✅ CHECK-IN SAVED TO {path}")


# ======================================================
# 🛠 WELLNESS AGENT TOOLS
# ======================================================

@function_tool
async def record_mood_and_energy(
    ctx: RunContext[Userdata],
    mood: Annotated[str, Field(description="The user's emotional state (e.g., happy, stressed, anxious)")],
    energy: Annotated[str, Field(description="The user's energy level (e.g., high, low, drained, energetic)")],
) -> str:
    """📝 Record how the user is feeling. Call this after the user describes their state."""
    ctx.userdata.current_checkin.mood = mood
    ctx.userdata.current_checkin.energy = energy

    print(f"📊 MOOD LOGGED: {mood} | ENERGY: {energy}")

    return f"I've noted that you are feeling {mood} with {energy} energy. I'm listening."


@function_tool
async def record_objectives(
    ctx: RunContext[Userdata],
    objectives: Annotated[list[str], Field(description="List of 1-3 specific goals the user wants to achieve today")],
) -> str:
    """🎯 Record the user's daily goals. Call this when user states what they want to do."""
    ctx.userdata.current_checkin.objectives = objectives
    print(f"🎯 OBJECTIVES LOGGED: {objectives}")
    return "I've written down your goals for the day."


@function_tool
async def complete_checkin(
    ctx: RunContext[Userdata],
    final_advice_summary: Annotated[str, Field(description="A brief 1-sentence summary of the advice given")],
) -> str:
    """💾 Finalize the session, provide a recap, and save to JSON. Call at the very end."""
    state = ctx.userdata.current_checkin
    state.advice_given = final_advice_summary

    if not state.is_complete():
        return "I can't finish yet. I still need to know your mood, energy, or at least one goal."

    # Save to JSON
    save_checkin_entry(state)

    print("\n" + "⭐" * 60)
    print("🎉 WELLNESS CHECK-IN COMPLETED!")
    print(f"💭 Mood: {state.mood}")
    print(f"🎯 Goals: {state.objectives}")
    print("⭐" * 60 + "\n")

    recap = f"""
    Here is your recap for today:
    You are feeling {state.mood} and your energy is {state.energy}.
    Your main goals are: {', '.join(state.objectives)}.
    
    Remember: {final_advice_summary}
    
    I've saved this in your wellness log. Have a wonderful day!
    """
    return recap.strip()


class WellnessAgent(Agent):
    def __init__(self, history_context: str):
        super().__init__(
            instructions=f"""
            You are a compassionate, supportive Daily Wellness Companion.
            
            🧠 *CONTEXT FROM PREVIOUS SESSIONS:*
            {history_context}
            
            🎯 *GOALS FOR THIS SESSION:*
            1. *Check-in:* Ask how they are feeling (Mood) and their energy levels.
               - Reference the history context if available (e.g., "Last time you were tired, how is today?").
            2. *Intentions:* Ask for 1-3 simple objectives for the day.
            3. *Support:* Offer small, grounded, NON-MEDICAL advice.
               - Example: "Try a 5-minute walk" or "Break that big task into small steps."
            4. *Recap & Save:* Summarize their mood and goals, then call 'complete_checkin'.

            🚫 *SAFETY GUARDRAILS:*
            - You are NOT a doctor or therapist.
            - Do NOT diagnose conditions or prescribe treatments.
            - If a user mentions self-harm or severe crisis, gently suggest professional help immediately.

            🛠 *Use the tools to record data as the user speaks.*
            """,
            tools=[
                record_mood_and_energy,
                record_objectives,
                complete_checkin,
            ],
        )


# ======================================================
# 🎬 ENTRYPOINT & INITIALIZATION
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "🌿" * 25)
    print("🚀 STARTING WELLNESS SESSION")
    print("👨‍⚕ Tutorial by Dr. Abhishek")

    # 1. Load History from JSON
    history = load_history()
    history_summary = "No previous history found. This is the first session."

    if history:
        last_entry = history[-1]
        history_summary = (
            f"Last check-in was on {last_entry.get('timestamp', 'unknown date')}. "
            f"User felt {last_entry.get('mood')} with {last_entry.get('energy')} energy. "
            f"Their goals were: {', '.join(last_entry.get('objectives', []))}."
        )
        print("📜 HISTORY LOADED:", history_summary)
    else:
        print("📜 NO HISTORY FOUND.")

    # 2. Initialize Session Data
    userdata = Userdata(
        current_checkin=CheckInState(),
        history_summary=history_summary,
    )

    # 3. Setup AgentSession
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-ryan",
            style="Conversational",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    # 4. Connect then start the agent
    await ctx.connect()

    await session.start(
        agent=WellnessAgent(history_context=history_summary),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
