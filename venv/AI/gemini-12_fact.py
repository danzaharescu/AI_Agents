import os
import json
import uuid
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

#create a virtual environment and activate it
#python -m venv venv
#.\venv\Scripts\Activate.ps1

# =====================================================================
# FACTOR 6 & 7: UNIFIED BUSINESS & EXECUTION STATE (Stateless Reducer)
# =====================================================================
class AgentState(BaseModel):
    session_id: str
    status: str = "RUNNING"  # RUNNING | WAITING_FOR_HUMAN | COMPLETED | FAILED
    step_count: int = 0
    max_steps: int = 5      # Hard ceiling on execution loop (Factor 5: Own Control Flow)
    messages: List[Dict[str, Any]] = []
    pending_tool_call: Optional[Dict[str, Any]] = None

# =====================================================================
# FACTOR 4: TOOLS ARE JUST STRUCTURED CODE
# =====================================================================
def get_user_balance(user_id: str) -> str:
    """Mock database lookup for user balance."""
    return f"User {user_id} balance is $1,250.00 USD."

def request_refund(user_id: str, amount: float) -> str:
    """High-stakes tool requiring human approval."""
    return f"Refund of ${amount} for user {user_id} successfully processed."

AVAILABLE_TOOLS = {
    "get_user_balance": get_user_balance,
    "request_refund": request_refund
}

SENSITIVE_TOOLS = {"request_refund"}  # Tools requiring human confirmation

# =====================================================================
# 12-FACTOR AGENT CLASS
# =====================================================================
class TwelveFactorAgent:
    def __init__(self, model_name: str = "gemini-3.1-flash-lite"):
        # Explicit SDK initialization (Uses GEMINI_API_KEY env var)
        self.client = genai.Client()
        self.model_name = model_name

    @staticmethod
    def _build_system_instruction() -> str:
        # FACTOR 2: OWN YOUR PROMPTS (No framework hidden prompts)
        return (
            "You are a helpful customer support agent. "
            "Always check user balance before issuing refunds. "
            "Limit actions strictly to provided tools."
        )

    def run(self, state: AgentState, user_input: Optional[str] = None) -> AgentState:
        """
        Pure state reducer: (Current State, Input) -> New State
        Allows pause, resume, and step-by-step auditability.
        """
        # Append user input if present
        if user_input:
            state.messages.append({"role": "user", "content": user_input})

        # FACTOR 5: OWN YOUR CONTROL FLOW (Explicit loop owned by Python, not LLM)
        while state.status == "RUNNING" and state.step_count < state.max_steps:
            state.step_count += 1
            print(f"\n--- [Step {state.step_count}] Invoking Gemini ---")

            # FACTOR 3: OWN YOUR CONTEXT WINDOW (Curate what enters history)
            gemini_contents = [
                types.Content(
                    role=msg["role"],
                    parts=[types.Part.from_text(text=msg["content"])]
                ) for msg in state.messages
            ]

            print
            for content in gemini_contents:
                # Join text parts in case a single message has multiple text blocks
                text = " ".join(part.text for part in content.parts if part.text)
                print(f"[{content.role.upper()}]: {text}")

            try:
                # FACTOR 1 & 4: NATURAL LANGUAGE TO STRUCTURED TOOL CALLS
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self._build_system_instruction(),
                        tools=[get_user_balance, request_refund],  # Native SDK function declaration,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),                        
                        temperature=0.1
                    )
                )

                # Check if model requested tool call(s)
                if response.function_calls:
                    call = response.function_calls[0]
                    tool_name = call.name
                    tool_args = call.args

                    print(f"🤖 LLM Requested Tool Call: `{tool_name}` with args {tool_args}")

                    # FACTOR 8: CONTACT HUMANS WITH TOOL CALLS (HITL Approval Gate)
                    if tool_name in SENSITIVE_TOOLS:
                        print("⚠️ SENSITIVE TOOL DETECTED: Pausing execution for human approval!")
                        state.status = "WAITING_FOR_HUMAN"
                        state.pending_tool_call = {
                            "name": tool_name,
                            "args": tool_args
                        }
                        return state  # Suspend execution state

                    # Execute non-sensitive tool immediately
                    result = self._execute_tool(tool_name, tool_args)
                    self._append_tool_result(state, tool_name, result)

                # Standard text response from model
                elif response.text:
                    print(f"🤖 Gemini: {response.text}")
                    state.messages.append({"role": "model", "content": response.text})
                    state.status = "COMPLETED"
                    return state

            # FACTOR 11: COMPACT ERRORS INTO CONTEXT
            except Exception as err:
                print(f"❌ Error encountered: {err}")
                error_msg = f"System Error: Execution failed during tool call. Details: {str(err)}"
                state.messages.append({"role": "user", "content": error_msg})

        if state.step_count >= state.max_steps and state.status == "RUNNING":
            print("🚨 Max execution steps reached. Stopping loop safely.")
            state.status = "FAILED"

        return state

    def resume_human_approval(self, state: AgentState, approved: bool) -> AgentState:
        """Resumes agent after human approval or rejection."""
        if state.status != "WAITING_FOR_HUMAN" or not state.pending_tool_call:
            raise ValueError("Agent is not currently awaiting human approval.")

        tool_call = state.pending_tool_call
        state.pending_tool_call = None

        if approved:
            print(f"✅ Human Approved tool `{tool_call['name']}`")
            result = self._execute_tool(tool_call["name"], tool_call["args"])
            self._append_tool_result(state, tool_call["name"], result)
            state.status = "RUNNING"
            return self.run(state)  # Re-enter control loop
        else:
            print(f"🛑 Human Rejected tool `{tool_call['name']}`")
            state.messages.append({
                "role": "user",
                "content": f"User rejected execution of tool '{tool_call['name']}'."
            })
            state.status = "RUNNING"
            return self.run(state)

    def _execute_tool(self, name: str, args: dict) -> str:
        fn = AVAILABLE_TOOLS.get(name)
        if not fn:
            return f"Error: Tool '{name}' not found."
        return fn(**args)

    def _append_tool_result(self, state: AgentState, name: str, result: str):
        print(f"🔧 Tool Output ({name}): {result}")
        state.messages.append({"role": "model", "content": f"[Tool Call: {name}]"})
        state.messages.append({"role": "user", "content": f"[Tool Output for {name}]: {result}"})


# =====================================================================
# DEMONSTRATION OF AGENT LIFECYCLE
# =====================================================================
if __name__ == "__main__":
    agent = TwelveFactorAgent(model_name="gemini-3.1-flash-lite")
    state = AgentState(session_id=str(uuid.uuid4())[:8])

    print(f"=== Support Agent Active (Session: {state.session_id}) ===")
    print("Type your message below. Enter 'exit' or 'quit' to quit.\n")

    while True:
        # 1. Handle Human-in-the-Loop Approval if agent paused on sensitive action
        if state.status == "WAITING_FOR_HUMAN":
            tool_info = state.pending_tool_call
            print(f"\n⏸️ ACTION REQUIRES APPROVAL: `{tool_info['name']}` with args {tool_info['args']}")
            
            decision = input("Approve execution? (yes/no): ").strip().lower()
            is_approved = decision in ["yes", "y"]
            
            # Resume execution flow (no new user_input needed)
            state = agent.resume_human_approval(state, approved=is_approved)
            continue

        # 2. Reset status if previous command completed, keeping state history intact
        if state.status in ["COMPLETED", "FAILED"]:
            state.status = "RUNNING"

        # 3. Prompt human for input
        try:
            user_input = input("\nUser > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session...")
            break

        # Check for exit command
        if user_input.lower() in ["exit", "quit"]:
            print("Session terminated. Goodbye!")
            break

        # Skip empty inputs
        if not user_input:
            continue

        # 4. Run the state reducer with user's input
        state = agent.run(state, user_input=user_input)



# How this code implements key 12-Factor principles:
# Own Your Control Flow (Factor 5): The while-loop, recursion limit (max_steps), and conditionals live in standard Python code rather than being hidden inside framework abstraction layers.
# Stateless Reducer (Factor 6 & 7): State is unified into the AgentState schema. You can save state.json to a database at any time, stop the process, and resume it hours later.
# Contact Humans via Tool Calls (Factor 8): Human approval isn't a side-channel script. The model attempts a tool call (request_refund), and our Python supervisor catches it, pauses execution, and waits for a structured human resume event.