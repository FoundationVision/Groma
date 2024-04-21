CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15
LOGDIR = "."

IGNORE_INDEX = -100
DEFAULT_TOKENS = {
    'pad': "[PAD]",
    'bos': "<s>",
    'eos': "</s>",
    'unk': "<unk>",
    'sep': "<sep>",
    'boi': "<img>",
    'eoi': "</img>",
    'bor': "<roi>",
    'eor': "</roi>",
    'boe': "<p>",
    'eoe': "</p>",
    'image': "<image>",
    'region': "<region>",
    'rbox': "<refer_box>",
    'gbox': "<ground_box>",
    'rfeat': "<refer_feat>",
    'ground': "[grounding]",
}
REGION_IDX_TOKENS = ['<r{}>'.format(i) for i in range(100)]
