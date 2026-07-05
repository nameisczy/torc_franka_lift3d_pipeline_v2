import argparse
import message_filters
import rospy
from std_msgs.msg import Int32, String


def run():
    parser = argparse.ArgumentParser(
        prog="show_selected_obj.sh",
        epilog="Shows information about the current selected object in MuJoCo.",
    )
    args = parser.parse_args()

    selected_id = 0
    selected_name = ""

    def ground_truth_info_cb(_selected_id: Int32, _selected_name: String):
        nonlocal selected_id, selected_name
        if _selected_id != selected_id and _selected_name != selected_name:
            print(f"  Selected: {str(_selected_id.data):<3} {_selected_name.data:<20}")
            selected_id = _selected_id
            selected_name = _selected_name

    rospy.init_node("show_selected_obj")
    ground_truth_info_sub = message_filters.ApproximateTimeSynchronizer(
        [
            message_filters.Subscriber("/ground_truth/selected_object_id", Int32),
            message_filters.Subscriber("/ground_truth/selected_object_name", String),
        ],
        10,
        0.1,
        allow_headerless=True,
    )
    ground_truth_info_sub.registerCallback(ground_truth_info_cb)

    print("👆 Showing selected objects...")

    try:
        rospy.spin()
    except:
        rospy.signal_shutdown()

    print("🛑 Exiting...")


if __name__ == "__main__":
    run()
