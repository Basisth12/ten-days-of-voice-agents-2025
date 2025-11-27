import logging
import os
import sqlite3
from datetime import datetime
from typing import Annotated, Optional
from dataclasses import dataclass

print("\n" + "🛡" * 50)
print("🚀 BANK FRAUD AGENT (SQLite) - INITIALIZED")
print("📚 TASKS: Verify Identity -> Check Transaction -> Update DB")
print("🛡" * 50 + "\n")

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
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO)
load_dotenv(".env.local")


DB_FILE = "fraud_db.sqlite"


@dataclass
class FraudCase:
    userName: str
    securityIdentifier: str
    cardEnding: str
    transactionName: str
    transactionAmount: str
    transactionTime: str
    transactionSource: str
    case_status: str = "pending_review"
    notes: str = ""


def get_db_path():
    """
    Return full path for DB. If __file__ is not present (REPL), fall back to current working dir.
    """
    base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    return os.path.join(base_dir, DB_FILE)


def get_conn():
    path = get_db_path()
    # Ensure directory exists (rarely necessary for script-run, but safe)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def seed_database():
    """Create SQLite DB and insert sample rows if empty."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fraud_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userName TEXT NOT NULL,
            securityIdentifier TEXT,
            cardEnding TEXT,
            transactionName TEXT,
            transactionAmount TEXT,
            transactionTime TEXT,
            transactionSource TEXT,
            case_status TEXT DEFAULT 'pending_review',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )

    cur.execute("SELECT COUNT(1) FROM fraud_cases")
    count = cur.fetchone()[0]
    if count == 0:
        sample_data = [
            (
                "John", "12345", "4242",
                "ABC Industry", "$450.00", "2:30 AM EST", "alibaba.com",
                "pending_review", "Automated flag: High value transaction."
            ),
            (
                "Sarah", "99887", "1199",
                "Unknown Crypto Exchange", "$2,100.00", "4:15 AM PST", "online_transfer",
                "pending_review", "Automated flag: Unusual location."
            )
        ]
        cur.executemany(
            """
            INSERT INTO fraud_cases (
                userName, securityIdentifier, cardEnding, transactionName,
                transactionAmount, transactionTime, transactionSource, case_status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            sample_data,
        )
        conn.commit()
        print(f"✅ SQLite DB seeded at {get_db_path()}")

    conn.close()


# Initialize DB on load
try:
    seed_database()
except Exception as e:
    logger.exception("Failed to seed database: %s", e)


@dataclass
class Userdata:
    active_case: Optional[FraudCase] = None


@function_tool
async def lookup_customer(
    ctx: RunContext[Userdata],
    name: Annotated[str, Field(description="The name the user provides")],
) -> str:
    """Lookup a customer in SQLite DB."""
    logger.info("🔎 LOOKING UP: %s", name)
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM fraud_cases WHERE LOWER(userName) = LOWER(?) LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        conn.close()

        if not row:
            return "User not found in the fraud database. Please repeat the name."

        record = dict(row)
        ctx.userdata.active_case = FraudCase(
            userName=record["userName"],
            securityIdentifier=record["securityIdentifier"],
            cardEnding=record["cardEnding"],
            transactionName=record["transactionName"],
            transactionAmount=record["transactionAmount"],
            transactionTime=record["transactionTime"],
            transactionSource=record["transactionSource"],
            case_status=record.get("case_status", "pending_review"),
            notes=record.get("notes", ""),
        )

        return (
            f"Record Found.\n"
            f"User: {record['userName']}\n"
            f"Security ID (Expected): {record['securityIdentifier']}\n"
            f"Transaction: {record['transactionAmount']} at {record['transactionName']} ({record['transactionSource']})\n"
            f"Ask user for their Security Identifier now."
        )

    except Exception as e:
        logger.exception("Database error in lookup_customer: %s", e)
        return f"Database error: {str(e)}"


@function_tool
async def resolve_fraud_case(
    ctx: RunContext[Userdata],
    status: Annotated[str, Field(description="confirmed_safe or confirmed_fraud")],
    notes: Annotated[str, Field(description="Notes on the user's confirmation")],
) -> str:

    if not ctx.userdata.active_case:
        return "Error: No active case selected."

    case = ctx.userdata.active_case
    case.case_status = status
    case.notes = notes

    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE fraud_cases
            SET case_status = ?, notes = ?, updated_at = datetime('now')
            WHERE userName = ?
            """,
            (case.case_status, case.notes, case.userName),
        )
        conn.commit()

        # Confirm updated row
        cur.execute("SELECT * FROM fraud_cases WHERE userName = ? LIMIT 1", (case.userName,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return "Error: Could not find the updated record."

        updated_row = dict(row)

        logger.info("✅ CASE UPDATED: %s -> %s", case.userName, status)

        if status == "confirmed_fraud":
            # (In reality you'd also call blocking API / trigger workflows)
            return (
                f"Fraud confirmed. Card ending {case.cardEnding} is now BLOCKED. "
                f"A replacement card will be issued.\n"
                f"DB Updated At: {updated_row.get('updated_at')}"
            )
        else:
            return (
                f"Transaction marked SAFE. Restrictions lifted.\n"
                f"DB Updated At: {updated_row.get('updated_at')}"
            )

    except Exception as e:
        logger.exception("Error saving to DB in resolve_fraud_case: %s", e)
        return f"Error saving to DB: {e}"


class FraudAgent(Agent):
    def __init__(self):
        # call base Agent initializer correctly
        super().__init__(
            instructions="""
            You are 'Alex', a Fraud Detection Specialist at Dishant Bank.
            Follow strict security protocol:

            1. Greeting + ask for first name.
            2. Immediately call lookup_customer(name).
            3. Ask for Security Identifier.
            4. If correct → continue. If incorrect → end call politely.
            5. Explain suspicious transaction.
            6. Ask: Did you make this transaction?
               - YES → resolve_fraud_case('confirmed_safe')
               - NO → resolve_fraud_case('confirmed_fraud')
            7. Close professionally.
            """,
            tools=[lookup_customer, resolve_fraud_case],
        )


def prewarm(proc: JobProcess):
    # Keep safe-guard: use try/except when loading models
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.exception("Failed to prewarm VAD; continuing without it.")
        proc.userdata["vad"] = None


async def entrypoint(ctx: JobContext):
    # Add safe logging of room name if available
    try:
        ctx.log_context_fields = {"room": getattr(ctx.room, "name", "unknown")}
    except Exception:
        ctx.log_context_fields = {}

    print("\n" + "💼" * 25)
    print("🚀 STARTING FRAUD ALERT SESSION (SQLite)")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    # Start the session with the configured agent
    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    # entrypoint and prewarm will be used by worker; run app
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
