import random
import sys

from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class FeedbackInterface(QWidget):
    def __init__(self, parent=None):
        super(FeedbackInterface, self).__init__(parent=parent)
        self.title = "Online Learning Feedback GUI"
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        self.setPalette(palette)

        self.choice1 = QRadioButton("Trajectory 1")
        self.choice1.setChecked(False)
        # self.choice1.toggled.connect(lambda:self.btnstate(self.choice1))

        self.choice2 = QRadioButton("Trajectory 2")
        self.choice2.setChecked(False)
        # self.choice2.toggled.connect(lambda:self.btnstate(self.choice2))

        self.choice3 = QRadioButton("No Preference")
        self.choice3.setChecked(True)
        # self.choice3.toggled.connect(lambda:self.btnstate(self.choice3))

        self.intro_text = QLabel(
            "Welcome to the feedback system for online learning. Using this system, you can give feedback"
            " to the robot. Please use the following keys on your keyboard to guide the robot using the appropriate method.",
            self,
        )
        self.navigation_label = QLabel(
            "   Navigation",
            self,
        )
        self.manipulation = QLabel(
            "                             Manipulation",
            self,
        )
        self.evaluative_feedback_label = QLabel(
            "Evaluative Feedback",
            self,
        )
        self.hierarchical_label = QLabel(
            "Hierarchical Imitation",
            self,
        )

        self.attention_feedback = QLabel(
            "Attention",
            self,
        )

        self.preference_feedback = QLabel(
            "Trajectory Preference",
            self,
        )

        self.robot_feedback = QLabel(
            "Robot Feedback",
            self,
        )

        self.corrective_advice_label = QLabel("Corrective Advice", self)
        self.corrective_advice_label.setStyleSheet("color: white; padding :20px")

        self.intro_text.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.intro_text.setWordWrap(True)

        self.corrective_advice_label.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Fixed
        )
        self.corrective_advice_label.setWordWrap(True)
        self.corrective_advice_label.setAlignment(QtCore.Qt.AlignCenter)

        self.navigation_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.navigation_label.setWordWrap(True)
        self.navigation_label.setAlignment(QtCore.Qt.AlignCenter)

        self.manipulation.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.manipulation.setWordWrap(True)
        self.manipulation.setAlignment(QtCore.Qt.AlignCenter)

        self.evaluative_feedback_label.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Fixed
        )
        self.evaluative_feedback_label.setWordWrap(True)
        self.evaluative_feedback_label.setAlignment(QtCore.Qt.AlignCenter)
        self.evaluative_feedback_label.setStyleSheet("color: white; padding :20px")

        self.hierarchical_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.hierarchical_label.setWordWrap(True)
        self.hierarchical_label.setAlignment(QtCore.Qt.AlignCenter)
        self.hierarchical_label.setStyleSheet("color: white; padding :20px")

        self.attention_feedback.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.attention_feedback.setWordWrap(True)
        self.attention_feedback.setAlignment(QtCore.Qt.AlignCenter)
        self.attention_feedback.setStyleSheet("color: white; padding :20px")

        self.preference_feedback.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.preference_feedback.setWordWrap(True)
        self.preference_feedback.setAlignment(QtCore.Qt.AlignCenter)
        self.preference_feedback.setStyleSheet("color: white; padding :20px")

        self.robot_feedback.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.robot_feedback.setWordWrap(True)
        self.robot_feedback.setAlignment(QtCore.Qt.AlignCenter)
        self.robot_feedback.setStyleSheet("color: white; padding :20px")

        self.loss_label = QLabel("0", self)
        self.reward_label = QLabel("0", self)
        self.bddl_goal_state = QLabel("0", self)
        self.mouse_x = QLabel("0", self)
        self.mouse_y = QLabel("0", self)
        self.selected_object = QLabel("0", self)
        self.state_estimation = QLabel("0", self)
        self.human_signal = QLabel("0", self)
        self.horizontalGroupBox = QGroupBox()

        self.w_button = QPushButton("W = +X")
        self.a_button = QPushButton("A = +Y")
        self.s_button = QPushButton("S = -X")
        self.d_button = QPushButton("D = -Y")
        self.z_button = QPushButton("Z = Z")
        self.x_button = QPushButton("X = -Z")
        self.one_button = QPushButton("1 = -1")
        self.two_button = QPushButton("2 = -2")
        self.prev_press = None

        self.createGridLayout()

        # call the gridlayout function
        self.loss_label.text = ""
        self.reward_label.text = ""
        # self.bddl_goal_state.text = ""
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.horizontalGroupBox)
        self.setLayout(windowLayout)
        self.show()  # this sets the main window to the screen size

    def createGridLayout(self):
        layout = QGridLayout()
        layout.addWidget(self.intro_text, 0, 0, 1, 6)

        rand_init_x = QLineEdit()
        rand_init_x.textChanged.connect(self.textchangedX)

        rand_init_y = QLineEdit()
        rand_init_y.textChanged.connect(self.textchangedY)

        rand_init_z = QLineEdit()
        rand_init_z.textChanged.connect(self.textchangedZ)

        load_model_num = QLineEdit()
        load_model_num.textChanged.connect(self.textChangedLoadModel)

        save_model_num = QLineEdit()
        save_model_num.textChanged.connect(self.textChangedLoadModel)

        layout.addWidget(QPushButton("R = Reset"), 1, 0)
        layout.addWidget(QPushButton("P = Pause"), 1, 1)
        layout.addWidget(QPushButton("Loss: "), 1, 2)
        layout.addWidget(self.loss_label, 1, 3)
        layout.addWidget(QPushButton("Reward: "), 1, 4)
        layout.addWidget(self.reward_label, 1, 5)

        layout.addWidget(QPushButton("Random Init X: "), 2, 0)
        layout.addWidget(rand_init_x, 2, 1)
        layout.addWidget(QPushButton("Random Init Y: "), 2, 2)
        layout.addWidget(rand_init_y, 2, 3)
        layout.addWidget(QPushButton("Random Init Z: "), 2, 4)
        layout.addWidget(rand_init_z, 2, 5)

        layout.addWidget(QPushButton("Load Model: "), 3, 1)
        layout.addWidget(load_model_num, 3, 2)
        layout.addWidget(QPushButton("Save Model: "), 3, 3)
        layout.addWidget(save_model_num, 3, 4)

        layout.addWidget(self.corrective_advice_label, 4, 3)
        layout.addWidget(self.navigation_label, 4, 1)
        layout.addWidget(QPushButton("E = Clockwise"), 6, 2)
        layout.addWidget(self.w_button, 6, 1)
        layout.addWidget(QPushButton("Q = Anti-Clockwise"), 6, 0)
        layout.addWidget(self.a_button, 7, 0)
        layout.addWidget(self.s_button, 7, 1)
        layout.addWidget(self.d_button, 7, 2)

        layout.addWidget(self.z_button, 8, 0)
        layout.addWidget(self.x_button, 8, 1)

        layout.addWidget(self.manipulation, 4, 4)
        layout.addWidget(QPushButton("O = Release Grip"), 6, 4)
        layout.addWidget(QPushButton("P = Contract Grip"), 6, 5)
        layout.addWidget(QPushButton("K = Extend Hand"), 7, 4)
        layout.addWidget(QPushButton("L = Retract Hand"), 7, 5)

        layout.addWidget(self.evaluative_feedback_label, 10, 0)

        layout.addWidget(self.one_button, 10, 1)
        layout.addWidget(self.two_button, 10, 2)
        layout.addWidget(QPushButton("I = Interrupt"), 10, 3)

        layout.addWidget(self.hierarchical_label, 12, 0)

        layout.addWidget(QPushButton("Current State: "), 12, 1)
        layout.addWidget(self.bddl_goal_state, 12, 2)

        bddl_options = QComboBox(self)
        bddl_options.addItem("not (open ?window.n.01_1)")
        bddl_options.addItem("not (open ?window.n.01_2)")
        bddl_options.addItem("not (open ?window.n.01_3)")
        bddl_options.addItem("not (open ?window.n.01_4)")

        layout.addWidget(QPushButton("Choose next BDDL: "), 12, 3)
        layout.addWidget(bddl_options, 12, 4)

        layout.addWidget(self.attention_feedback, 15, 0)

        layout.addWidget(QPushButton("Mouse X: "), 15, 1)
        layout.addWidget(self.mouse_x, 15, 2, Qt.AlignCenter)
        layout.addWidget(QPushButton("Mouse Y: "), 15, 3)
        layout.addWidget(self.mouse_y, 15, 4, Qt.AlignCenter)
        layout.addWidget(QPushButton("Selected Object: "), 15, 5)
        layout.addWidget(self.selected_object, 15, 6)

        layout.addWidget(self.preference_feedback, 18, 0)

        layout.addWidget(QPushButton("I prefer: "), 18, 1)
        layout.addWidget(self.choice1, 18, 2)
        layout.addWidget(self.choice2, 18, 3)
        layout.addWidget(self.choice3, 18, 4)

        layout.addWidget(self.robot_feedback, 21, 0)
        layout.addWidget(QPushButton("Uncertainty in state estimation: "), 21, 1)
        layout.addWidget(self.state_estimation, 21, 2, Qt.AlignCenter)
        layout.addWidget(QPushButton("Uncertainty in predicting human signal: "), 21, 4)
        layout.addWidget(self.human_signal, 21, 5, Qt.AlignCenter)

        self.horizontalGroupBox.setLayout(layout)

    def btnstate(self, btn):
        btn.setChecked(not btn.isChecked())

    def updateLoss(self, loss_val):
        self.loss_label.setText("{:.5}".format(str(loss_val)))

    def updateMouseX(self, mouseX):
        self.mouse_x.setText(str(mouseX))

    def updateMouseY(self, mouseY):
        self.mouse_y.setText(str(mouseY))

    def updateSelectedObject(self, selected_object):
        self.selected_object.setText(selected_object)

    def updateReward(self, reward_val):
        self.reward_label.setText("{:.5}".format(str(reward_val)))

    def updateStateEstimation(self, state):
        self.state_estimation.setText("{:.5}".format(str(state)))

    def updateHumanSignal(self, signal):
        self.human_signal.setText("{:.5}".format(str(signal)))

    def updateBDDL(self, bddl_val):
        self.bddl_goal_state.setText(str(bddl_val))

    def textchangedX(self, text):
        self.X = text

    def textchangedY(self, text):
        self.Y = text

    def textchangedZ(self, text):
        self.Z = text

    def textChangedLoadModel(self, text):
        self.LoadModel = text

    def update_font(self, key):

        button_map = {
            "'w'": self.w_button,
            "'a'": self.a_button,
            "'s'": self.s_button,
            "'d'": self.d_button,
            "'z'": self.z_button,
            "'x'": self.x_button,
            "'1'": self.one_button,
            "'2'": self.two_button
        }
        if str(key) in button_map:
            if self.prev_press:
                self.prev_press.setStyleSheet("color: black")
            button_map[str(key)].setStyleSheet("color: red")
            self.prev_press = button_map[str(key)]
        else:
            print(key)


def main():
    app = QApplication(sys.argv)
    ex = FeedbackInterface()
    ex.updateLoss(random.randint(0, 1000) / 1000.0)
    ex.updateReward(random.random() * 100)
    ex.updateBDDL("not (open ?window.n.01_1)")
    ex.updateMouseX("156")
    ex.updateMouseY("612")
    ex.updateSelectedObject("Chair")
    ex.updateStateEstimation(random.random() * 100)
    ex.updateHumanSignal(random.random() * 100)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
