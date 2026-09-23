from ai_backend import TextAI
from message_handling import MessageHandler

# Create a new instance of the TextAI class
text_ai = TextAI()
text_ai.backend.set_default("chat",model="gpt-4")

# Create a new instance of the MessageHandler class
messages = MessageHandler()

message1 = "Hello, this is an API test. Please response with 'Message received'."
messages.continue_messages("user", message1)

response = text_ai.text_chat(messages.messages)

print(messages.messages)
print(response)