import json

from backend.model.rdb_persistence.persistent import PersistableCommand
from backend.commands.specification.dummy_command import DummyCommand
from backend.commands.specification.metadata_command import MetadataCommand


"""
- Parser
  - Receives a stream which contains one or more commands, creates one or more instances of commands

- CommandsExecutor
  - Receives a Workspace and one or more command instances to be executed

- Command.
  - Can be created directly, by a Parser (different Parsers possible), or by deserialization
  - Serializable/deserializable in: JSON or DataFrame format. Execute. Acting always over a
  - Could have a method to estimate the execution time

- Workspace
  - Can be serialized/deserialized
  - Allows reading, writing, deleting variables
  - Can integrate the result of a Command (if the Command does not have this capability)

"""


def generator_primitive_json(input):
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


def command_generator_parser_factory(generator_type, file_type, file):
    """
    Returns a generator appropriate to parse "file" and generate commands

    :param generator_type: 
    :param file_type: 
    :param file: 
    :return: 
    """
    # TODO Prepare the input stream. It can be a String, a URL, a file, a Stream
    # TODO Define a routine to read it into memory
    s = file
    if generator_type.lower() in ["rscript", "r-script", "r"]:
        # TODO The R script was prepared to be run from outside NIS, using R NIS client
        # TODO Running the script from the inside should be managed slightly different:
        # TODO - Recognize that it is an internally launched script
        # TODO   - The R script will open an interactive session: do not open a new InteractiveSession and
        # TODO   - find a way to reenter the launching Int.Sess.
        # TODO   - ignore open/close session commands creating or saving case studies
        # TODO   - execute commands modifying in memory state, ignore others
        #
        # TODO Take the R script and launch it as a separate process.
        # TODO
        pass
    elif generator_type.lower() in ["python", "python-script"]:
        # TODO Exact same considerations as for R scripts
        pass
    elif generator_type.lower() in ["spreadsheet", "excel", "workbook"]:
        # TODO A sequence of commands, providing the whole case study
        pass
    elif generator_type.lower() in ["json", "primitive"]:
        # A list of commands. Each command is a dictionary: the command type, a label and the content
        # The type is for the factory to determine the class to instantiate, while label and content
        # are passed to the command constructor to elaborate the command
        if isinstance(s, bytes):
            s = s.decode("utf-8")
        yield from generator_primitive_json(s)


def execute_command(state, cmd: "IExecutableCommand"):
    return cmd.execute(state)


def execute_command_generator(state, cmd: PersistableCommand):
    return execute_command_generator_file(state, cmd.generator_type, cmd.content_type, cmd.content)


def execute_command_generator_file(state, generator_type, file_type: str, file):
    """
    Creates a generator parser, then it feeds the file type and the file
    The generator parser has to parse the file and to generate commands as a Python generator 

    :param generator_type: 
    :param file_type: 
    :param file: 
    :return: Issues and outputs (still not defined) -TODO- 
    """
    # Generator factory (from generator_type and file_type)
    # Generator has to call "yield" whenever an ICommand is generated
    for cmd in command_generator_parser_factory(generator_type, file_type, file):
        execute_command(state, cmd)
    # TODO Compile (accumulate) issues and outputs
    return None


def create_command(cmd_type, name, json_input):
    """
    Factory creating and initializing a command from its type, optional name and parameters

    :param cmd_type: String describing the type of command, as found in the interchange JSON format
    :param name: An optional name (label) for the command
    :param json_input: Parameters specific of the command type (each command knows how to interpret,
                       validate and integrate them)
    :return: The instance of the command
    :raise
    """
    cmds = {"dummy":    DummyCommand,
            "metadata": MetadataCommand}
    if cmd_type in cmds:
        tmp = cmds[cmd_type](name)
        tmp._serialization_type = cmd_type  # Injected attribute. Used later for serialization
        tmp._serialization_label = name  # Injected attribute. Used later for serialization
        if isinstance(json_input, str):
            tmp.json_deserialize(json_input)
        elif isinstance(json_input, dict):
            tmp.json_deserialize(json.dumps(json_input))
        else:
            # NO SPECIFICATION
            raise Exception("The command '" + cmd_type + " " + name if name else "<unnamed>" + " does not have a specification.")
        return tmp
    else:
        # UNKNOWN COMMAND
        raise Exception("Unknown command type: " + cmd_type)

if __name__ == '__main__':
    def gen(n):
        def g2(g):
            yield 2*g

        i = 0
        while i<n:
            yield from g2(i)
            i+=1
    for r in gen(5):
        print(str(r))