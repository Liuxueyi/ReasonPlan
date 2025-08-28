CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

MAP_TOKEN_INDEX = -201
DEFAULT_MAP_TOKEN = "<map>"

PERCEPTION_TOKEN_INDEX = -202
DEFAULT_PERCEPTION_TOKEN = "<perception>"

OBJ_TOKEN_INDEX = -203
DEFAULT_OBJ_TOKEN = "<obj>"

NAVI_TOKEN_INDEX = -204
DEFAULT_NAVI_TOKEN = "<navigation>"

CMD_TOKEN_INDEX = -205
DEFAULT_CMD_TOKEN = "<command>"

EGO_TRAJ_TOKEN_INDEX = -300
DEFAULT_EGO_TRAJ_TOKEN = "<ego_traj>"



# Prompts
TEXT_PROMPT = \
"You are the brain of an autonomous vehicle. \
You're at point (0,0). X-axis is perpendicular and Y-axis is parallel to the direction you're facing.\n"

TEXT_INPUT_HEAD = "\nInput: \n"

TEXT_INPUT_CLIP_IMG = "- Front view figure <image>.\n" # TODO maybe move <image> to the begin of the text
TEXT_INPUT_MAP = "- Road structure information centered on ego-vehicle <map>.\n"
TEXT_INPUT_PERCEPTION = "- Perception information <perception>.\n"
TEXT_INPUT_OBJ = "- Object detection information <obj>.\n"
TEXT_INPUT_NAVI = "- Navigation information <navigation>.\n"
TEXT_INPUT_CMD = "- Command information <command>.\n"

TEXT_TASK = \
"\nTask: Plan a safe and feasible 8-second driving trajectory at 2 Hz with 16 waypoints. \
Avoid collisions with other objects.\n"

TEXT_ANSWER_EGO_TRAJ = "- Ego-vehicle trajectory prediction <ego_traj>.\n"

