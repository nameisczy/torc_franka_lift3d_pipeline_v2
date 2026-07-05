# Parse rosbag
import argparse
from typing import Any, Callable, List, Optional, Tuple
import rosbag


def get_fields_str_recursive(
    obj: Any, indent=0, field_condition: Callable[[str, Any], bool] = lambda f, v: True
):
    """
    Recurisvely gets an object (obj)'s fields in the format `field: type = value`
    Only prints field value if it is a primitive type (str, int, float, bool).
    """
    print_strs = []

    def helper(obj: Any, indent=0) -> bool:
        fields = [x for x in dir(obj) if not x.startswith("_")]
        # sort fields by properties first, then methods
        fields.sort(key=lambda x: (callable(getattr(obj, x)), x))

        def lprint(add: bool, msg: str):
            nonlocal print_strs
            if add:
                print_strs.append(msg)

        any_field_cond_satisfied = False

        for field in fields:
            if field.startswith(("_", "deserialize", "serialize")):
                continue
            value = getattr(obj, field)
            field_cond_satisfied = field_condition(field, value)

            any_field_cond_satisfied = any_field_cond_satisfied or field_cond_satisfied

            if isinstance(value, (str, int, float, bool)):
                # Print primitive
                lprint(
                    field_cond_satisfied,
                    f"{' ' * indent}{field}: {type(value)} = {value}",
                )
            elif callable(value):
                # Print method
                # print parameters of callable value
                params = ", ".join(
                    getattr(value, "__code__").co_varnames[
                        : getattr(value, "__code__").co_argcount
                    ]
                )
                lprint(field_cond_satisfied, f"{' ' * indent}{field}({params})")
            elif hasattr(value, "__slots__"):
                # Recurse object
                sub_cond_satisfied = helper(value, indent + 2)
                lprint(
                    field_cond_satisfied or sub_cond_satisfied,
                    f"{' ' * indent}{field}: {type(value)}",
                )
                any_field_cond_satisfied = (
                    any_field_cond_satisfied or sub_cond_satisfied
                )
            elif (
                isinstance(value, (list, set))
                and len(value) > 0
                and hasattr(value[0], "__slots__")
            ):
                # Recurse list of objects
                for i, item in enumerate(value):
                    field_cond_satisfied = field_condition(f"{field}[{i}]", item)
                    sub_cond_satisfied = helper(item, indent + 4)
                    lprint(
                        field_cond_satisfied or sub_cond_satisfied,
                        f"{' ' * (indent + 2)}[{i}]: {type(item)}",
                    )
                    any_field_cond_satisfied = (
                        any_field_cond_satisfied or sub_cond_satisfied
                    )
            else:
                # Print any other type (e.g. object, list)
                lprint(field_cond_satisfied, f"{' ' * indent}{field}: {type(value)}")

        return any_field_cond_satisfied

    helper(obj, indent)

    return "\n".join(reversed(print_strs))


def parse_rosbag(rosbag_path: str):
    bag = rosbag.Bag(rosbag_path)

    count = 0
    for topic, msg, t in bag.read_messages():

        def condition(field: str, value):
            return isinstance(value, str) and "motoman" in value

        field_str = get_fields_str_recursive(msg, indent=2, field_condition=condition)
        if len(field_str.strip()) > 0:
            print()
            print(f"topic: {topic} msg: {type(msg)} timestamp: {t.to_sec()}")
            print(field_str)

        count += 1

    bag.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ROS bag to JSON format.")
    parser.add_argument(
        "--rosbag",
        "-r",
        type=str,
        required=True,
        help="Path to the input ROS bag file.",
    )
    # parser.add_argument(
    #     "--output", "-o", type=str, required=True, help="Path to the output JSON file."
    # )

    args = parser.parse_args()

    parse_rosbag(args.rosbag)
