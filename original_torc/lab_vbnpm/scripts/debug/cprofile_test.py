import cProfile
import pstats
from io import StringIO


def my_func() -> int:
    num = 0
    for i in range(1000000):
        num += i
    return num


if __name__ == "__main__":

    pr = cProfile.Profile()
    pr.enable()
    my_func()
    pr.disable()

    pr.dump_stats("my_func_profile.prof")

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    print(s.getvalue())
