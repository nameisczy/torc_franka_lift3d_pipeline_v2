def printColor(code, *args):
    print("\033[{}m{}\033[00m".format(code, " ".join(map(str, args))), flush=True)


def printRed(*args):  # print("\033[91m{}\033[00m" .format(args))
    printColor(91, *args)


def printPink(*args):  # print("\033[95m{}\033[00m" .format(args))
    printColor(95, *args)


def printGreen(*args):  # print("\033[92m{}\033[00m" .format(args))
    printColor(92, *args)


def printYellow(*args):  # print("\033[93m{}\033[00m" .format(args))
    printColor(93, *args)


def printLightPurple(*args):  # print("\033[94m{}\033[00m" .format(args))
    printColor(94, *args)


def printPurple(*args):  # print("\033[95m{}\033[00m" .format(args))
    printColor(95, *args)


def printBlue(*args):  # print("\033[34m{}\033[00m" .format(args))
    printColor(34, *args)


def printCyan(*args):  # print("\033[96m{}\033[00m" .format(args))
    printColor(96, *args)


def printLightGray(*args):  # print("\033[97m{}\033[00m" .format(args))
    printColor(97, *args)


def printBlack(args):  # print("\033[98m{}\033[00m" .format(args))
    printColor(98, *args)