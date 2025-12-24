import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from message_handling import call_tts
from utils import generate_file_location, load_config
from openai import OpenAI

text_input = "In the twilight’s gentle embrace, a solitary figure seeks the quiet company of nature. A writer, her heart heavy with unformed words, finds respite where the grass dances softly with the breeze. Her touch upon the earth is tender—a silent plea for inspiration.\
Beside her, a fox, the forest’s own storyteller, mirrors her contemplative stillness. Together, they share the sacred silence of the grove, an unspoken pact between kindred spirits.\
Around them, ancient trees stand as steadfast guardians, their leaves whispering secrets of endurance through the ages. The lake, a canvas reflecting the day’s last light, cradles both history and the present moment, a serene witness to the cycles of the creative spirit.\
Here, in this fleeting sanctuary, the writer and the wild find a shared solace, a momentary reprieve from the world’s relentless march. It’s a reminder that in the quiet spaces, we find clarity, and in solitude, a universe of possibilities whispers back."

file_name = "The Whispering Wilderness"

config, _cfg_meta = load_config()
client = OpenAI(api_key=config['openai']['key'])
save_location_and_name = generate_file_location(config['image']['save_path'], file_name, '.mp3')
print(save_location_and_name)
call_tts(client,text_input,save_location_and_name,model_name="tts-1-hd",voice_type="fable")
