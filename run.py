import argparse

from zamr.cmds import Train, Predict, Evaluate, Preprocess

if __name__ == '__main__':
    # TODO: Release version delete all uselese code
    # TODO: 1. unused parameters: selection rate cgcn
    # TODO: 2. unused code:rec_gcn, cgcn, selection loss
    # TODO: refactor: rec -> refine

    cmds = argparse.ArgumentParser(description="Run ZAmr with commands (train, predict, evaluate, preprocess)")
    sub_cmd = cmds.add_subparsers(title='Commands', dest='mode')
    sub_commands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train(),
        'preprocess': Preprocess(),
    }

    # Add arguments for sub commands.
    for name, sub_command in sub_commands.items():
        sub_parser = sub_command.add_subparser(name, sub_cmd)

    args = cmds.parse_args()
    print(f"Run the {args.mode} command...")
    # run the commands
    cmd = sub_commands[args.mode]
    cmd(args)
