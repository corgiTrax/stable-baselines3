import threading as th

from pynput import keyboard, mouse

from stable_baselines3.common.corrective_feedback import behavior_robot, fetch_robot


class HumanFeedback:
    def __init__(self, robot=None, feedback_gui=None):
        super(HumanFeedback, self).__init__()
        self.keyboard_feedback_dictionary = {
            keyboard.KeyCode.from_char("1"): -10,
            keyboard.KeyCode.from_char("2"): 10,
        }

        if robot:
            if "FetchGripper" in robot:
                for key in fetch_robot:
                    self.keyboard_feedback_dictionary[key] = fetch_robot[key]
            if "BehaviorRobot" in robot:
                for key in behavior_robot:
                    self.keyboard_feedback_dictionary[key] = behavior_robot[key]

        self.keyboard_control_dictionary = {
            keyboard.KeyCode.from_char("p"): "Pause",
            keyboard.KeyCode.from_char("r"): "Reset",
        }
        self.human_keyboard_feedback = None
        self.run_keyboard_capture_thread()

        self.human_mouse_feedback = None
        self.start_mouse_thread = False
        self.run_mouse_capture_thread()

        self.feedback_gui = feedback_gui

    def keyboard_capture_thread(self):
        with keyboard.Events() as events:
            event = events.get(1e6)
            self.human_keyboard_feedback = event

    def run_keyboard_capture_thread(self):
        th.Thread(
            target=self.keyboard_capture_thread,
            args=(),
            name="keyboard_capture_thread",
            daemon=True,
        ).start()

    def return_human_keyboard_feedback(self):
        feedback = None
        if self.human_keyboard_feedback:
            if "Press" in str(
                self.human_keyboard_feedback
            ):  # only use keypresses as reward signals
                if (
                    self.human_keyboard_feedback.key
                    in self.keyboard_feedback_dictionary
                ):
                    feedback = self.keyboard_feedback_dictionary[
                        self.human_keyboard_feedback.key
                    ]
                    if self.feedback_gui:
                        self.feedback_gui.update_font(self.human_keyboard_feedback.key)
                elif (
                    self.human_keyboard_feedback.key in self.keyboard_control_dictionary
                ):  # use 'p' for pausing:
                    feedback = self.keyboard_control_dictionary[
                        self.human_keyboard_feedback.key
                    ]

            self.run_keyboard_capture_thread()

        self.human_keyboard_feedback = None
        return feedback

    def mouse_capture_thread(self):
        with mouse.Events() as events:
            event = events.get(1e6)
            self.start_mouse_thread = True
            if "Click" in str(event) and "pressed=False" in str(event):
                self.human_mouse_feedback = event

    def run_mouse_capture_thread(self):
        th.Thread(
            target=self.mouse_capture_thread,
            args=(),
            name="mouse_capture_thread",
            daemon=True,
        ).start()

    def return_human_mouse_feedback(self):
        feedback = None
        if self.human_mouse_feedback:
            feedback = self.human_mouse_feedback

        if self.start_mouse_thread:
            self.run_mouse_capture_thread()
            self.start_mouse_thread = False

        self.human_mouse_feedback = None
        return feedback
