"""launch(): print the splash, load the GUI stack, run TURT."""

import sys


def launch(argv=None):
    from .banner import print_environment, print_splash

    argv = sys.argv if argv is None else argv
    print_splash()
    print("   loading Py-ART / matplotlib ...", flush=True)

    from .app import main  # heavy imports happen here
    print_environment()
    print("   starting TURT -- good luck out there.\n", flush=True)
    main(argv[1] if len(argv) > 1 else None)
