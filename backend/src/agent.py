import logging
import json
import os
import asyncio
from typing import Annotated, Literal, Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path

print("\n" + "🧬" * 50)
print("🚀 Soft Computing Tutor")
print("💡 agent.py LOADED SUCCESSFULLY!")
print("🧬" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field

# NOTE: these imports are kept as in your original file.
# Ensure the packages providing these modules are installed in your environment.
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

# 🔌 PLUGINS (keep these if you have them installed)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")  # loads environment overrides from .env.local if present

# 🆕 file generation / path safety
CONTENT_FILE = "soft_computing.json"

DEFAULT_CONTENT = [
    {
        "id": "fuzzy_logic",
        "title": "The Fuzzy Logic",
        "summary": "Fuzzy logic is a mathematical framework for dealing with uncertainty and approximate reasoning, allowing values between true and false.",
        "sample_question": "What is fuzzy logic and how does it differ from classical binary logic?"
    },
    {
        "id": "neural_networks",
        "title": "The Neural Networks",
        "summary": "Neural networks are computational models inspired by the human brain, used in pattern recognition and machine learning.",
        "sample_question": "Describe how a neural network learns from data."
    },
    {
        "id": "genetic_algorithms",
        "title": "The Genetic Algorithms",
        "summary": "Genetic algorithms are optimization techniques inspired by the process of natural selection, used to solve complex problems.",
        "sample_question": "What are the main steps in a genetic algorithm?"
    },
    {
        "id": "soft_computing_features",
        "title": "The Soft Computing Features",
        "summary": "Soft Computing deals with approximate solutions, tolerance for imprecision, and the ability to handle uncertainty, unlike traditional hard computing.",
        "sample_question": "List some key differences between soft computing and hard computing."
    }
]


def load_content() -> List[Dict[str, str]]:
    """
    📖 Checks if soft computing JSON exists.
    If NO: Generates it from DEFAULT_CONTENT.
    If YES: Loads it.
    Uses pathlib for robust path handling; falls back to CWD if __file__ not defined.
    """
    try:
        # Determine base directory (safe if running from REPL where __file__ may be absent)
        try:
            base_dir = Path(__file__).parent
        except NameError:
            base_dir = Path.cwd()

        path = base_dir / CONTENT_FILE

        # Create file with default content if it doesn't exist
        if not path.exists():
            print(f"⚠ {CONTENT_FILE} not found at {path}. Generating soft computing data...")
            with path.open("w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONTENT, f, indent=4, ensure_ascii=False)
            print(f"✅ Soft computing data created at {path}")

        # Read and return data
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Content file must contain a list of topic dicts.")
            return data

    except Exception as e:
        logger.exception("Error managing content file")
        print(f"⚠ Error managing content file: {e}")
        return []


# Load data immediately on startup
COURSE_CONTENT: List[Dict[str, str]] = load_content()


@dataclass
class TutorState:
    """🧠 Tracks the current learning context"""
    current_topic_id: Optional[str] = None
    current_topic_data: Optional[dict] = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"

    def set_topic(self, topic_id: str) -> bool:
        # Find topic in loaded content (case-insensitive match)
        topic = next((item for item in COURSE_CONTENT if item["id"].lower() == topic_id.lower()), None)
        if topic:
            self.current_topic_id = topic["id"]
            self.current_topic_data = topic
            return True
        return False


@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None


@function_tool
async def select_topic(
    ctx: RunContext[Userdata],
    topic_id: Annotated[str, Field(description="The ID of the topic to study (e.g., 'fuzzy_logic')")]
) -> str:
    """📚 Selects a topic to study from the available list."""
    state = ctx.userdata.tutor_state
    success = state.set_topic(topic_id)
    if success:
        return f"Topic set to '{state.current_topic_data['title']}'. Ask the user if they want to 'Learn', be 'Quizzed', or 'Teach it back'."
    else:
        available = ", ".join([t["id"] for t in COURSE_CONTENT])
        return f"Topic not found. Available topics are: {available}"


@function_tool
async def set_learning_mode(
    ctx: RunContext[Userdata],
    mode: Annotated[str, Field(description="The mode to switch to: 'learn', 'quiz', or 'teach_back'")]
) -> str:
    """🔄 Switches the interaction mode and updates the agent's voice/persona."""
    state = ctx.userdata.tutor_state
    state.mode = mode.lower()

    agent_session = ctx.userdata.agent_session

    instruction = ""
    if agent_session:
        try:
            if state.mode == "learn":
                # 👨‍🏫 MATTHEW: The Lecturer
                agent_session.tts.update_options(voice="en-US-matthew", style="Promo")
                instruction = f"Mode: LEARN. Explain: {state.current_topic_data.get('summary', '')}"

            elif state.mode == "quiz":
                # 👩‍🏫 ALICIA: The Examiner
                agent_session.tts.update_options(voice="en-US-alicia", style="Conversational")
                instruction = f"Mode: QUIZ. Ask this question: {state.current_topic_data.get('sample_question', '')}"

            elif state.mode == "teach_back":
                # 👨‍🎓 KEN: The Student/Coach
                agent_session.tts.update_options(voice="en-US-ken", style="Promo")
                instruction = "Mode: TEACH_BACK. Ask the user to explain the concept to you as if YOU are the beginner."
            else:
                return "Invalid mode."
        except Exception as e:
            logger.exception("Failed to update tts options")
            instruction = f"Mode set to {state.mode}, but failed to change voice: {e}"
    else:
        instruction = "Voice switch failed (Session not found). Mode set locally."

    print(f"🔄 SWITCHING MODE -> {state.mode.upper()}")
    return f"Switched to {state.mode} mode. {instruction}"


@function_tool
async def evaluate_teaching(
    ctx: RunContext[Userdata],
    user_explanation: Annotated[str, Field(description="The explanation given by the user during teach-back")]
) -> str:
    """
    📝 Very simple evaluator: checks presence of some keywords from the topic summary
    and returns a score out of 10 plus short feedback. This is deterministic and
    intentionally simple so it can run locally without external models.
    """
    print(f"📝 EVALUATING EXPLANATION: {user_explanation}")

    state = ctx.userdata.tutor_state
    if not state.current_topic_data:
        return "No topic selected. Please select a topic before evaluating."

    summary = state.current_topic_data.get("summary", "")
    # extract keywords (naive)
    keywords = [w.lower().strip(".,") for w in summary.split() if len(w) > 4]
    # deduplicate and limit
    seen = []
    for k in keywords:
        if k not in seen:
            seen.append(k)
    keywords = seen[:6]

    found = sum(1 for kw in keywords if kw in user_explanation.lower())
    # Score: proportion of keywords found -> scaled to 10
    score = int(round((found / max(1, len(keywords))) * 10))
    feedback = []

    if score >= 8:
        feedback.append("Excellent; strong accuracy and clarity.")
    elif score >= 5:
        feedback.append("Good — covers major points but misses some detail.")
    else:
        feedback.append("Needs improvement — important points missing or unclear.")

    feedback.append(f"Keywords checked: {keywords}")
    feedback.append(f"Found {found} / {len(keywords)} keywords. Score: {score}/10")

    return " ".join(feedback)


class TutorAgent(Agent):
    def __init__(self):
        # Use super().__init__ (correct dunder) and pass instructions/tools
        super().__init__(
            instructions="""
            You are a Soft Computing Tutor. Help users learn about fuzzy logic, neural networks, and genetic algorithms.

            🎯 *YOUR ROLE:*
            - Ask what topic they want to study
            - Use tools to select topics and switch learning modes
            - Provide clear, engaging explanations

            📚 *MODES:*
            - LEARN: Explain concepts clearly
            - QUIZ: Test their knowledge
            - TEACH_BACK: Let them explain to you
            """,
            tools=[select_topic, set_learning_mode, evaluate_teaching],
        )


def prewarm(proc: JobProcess):
    # Preload voice activity detection (if plugin available)
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.exception("Failed to prewarm VAD; continuing without prewarm.")
        proc.userdata["vad"] = None


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "🧬" * 25)
    print("🚀 STARTING SOFT COMPUTING SESSION")
    print(f"📚 Loaded {len(COURSE_CONTENT)} topics from Knowledge Base")

    # 1. Initialize State
    userdata = Userdata(tutor_state=TutorState())

    # 2. Setup AgentSession (ensure these plugin classes exist in your environment)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Promo",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    # 3. Store session in userdata for tools to access
    userdata.agent_session = session

    # 4. Start
    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    # Cli runner — ensure WorkerOptions & cli.run_app exist in your livekit sdk
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
