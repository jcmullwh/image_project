import copy

class MessageHandler():

    def __init__(self, system_prompt=None):

        if system_prompt:
            self.messages = self.generate_first_message(system_prompt)
        else:
            default_prompt = "You are a helpful AI assistant."
            self.messages = self.generate_first_message(default_prompt)

    def generate_first_message(self,system_prompt):
        message = []
        message = [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                    ]
        
        return message

    def continue_messages(self,role,new_prompt):
        
        self.messages.append({
            "role": role,
            "content":new_prompt
            })
    
    def copy(self):
        return copy.deepcopy(self)