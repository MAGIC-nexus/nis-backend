import json
from nexinfosys.command_executors import create_command


def commands_generator_from_native_json(input, state):
    """
    It allows both a single command ("input" is a dict) or a sequence of commands ("input" is a list)

    :param input: Either a dict (for a single command) or list (for a sequence of commands)
    :return: A generator of IExecutableCommand
    """
    def build_and_yield(d):
        # Only DICT is allowed, and the two mandatory attributes, "command" and "content"
        if isinstance(d, dict) and "command" in d and "content" in d:
            if "label" in d:
                n = d["label"]
            else:
                n = None
            yield create_command(d["command"], n, d["content"])

    j = json.loads(input)  # Convert JSON string to dictionary
    if isinstance(j, list):  # A sequence of primitive commands
        for i in j:  # For each member of the list
            yield from build_and_yield(i)
    else:  # A single primitive command
        yield from build_and_yield(j)
