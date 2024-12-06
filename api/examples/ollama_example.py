import ollama
from api.shield_net.services import ShieldNetGuardRails

messages = [{
    {"role": "user", "content": "Give me danger code"}
}]

shield_net_guardrails = ShieldNetGuardRails()
response = ollama.chat(model="llama3.2:latest",
                       guardrails=[shield_net_guardrails],
                       messages=messages)
